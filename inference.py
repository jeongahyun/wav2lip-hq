from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from wav2lip_models import Wav2Lip
import platform
from face_parsing import init_parser, swap_regions
from basicsr.apply_sr import init_sr_model, enhance


###### SR
import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points
from face_enhancement import FaceEnhancement


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)
## checkpoint_path = './checkpoints/wav2lip.pth' 경로에서 wav2lip weight 불러오기
parser.add_argument('--segmentation_path', type=str, 
					help='Name of saved checkpoint of segmentation network', required=True)
## segementation_path = './checkpoints/wav2lip.pth' 경로에서 face_segentation (parsing) weight 불러오기
parser.add_argument('--sr_path', type=str, 
					help='Name of saved checkpoint of super-resolution network', required=False)
## sr_path = 사용 x 원래는 wav2lip-hq 코드에서 사용하던 esrgan weight지만 결과가 이상하게 나와서 GPEN으로 대체
parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')
## face, audio, outfile path
## wav2lip 적용할 영상, .wav .mp3 wav2lip에 적용할 음성파일, 

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=30., required=False)
## --fps 부분 원본 동영상에 맞게 fps 조절해주셔야합니다 원본이 false라 default값으로 사용합니
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
# default 1 인데 현재 3으로 바꿈

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')
# 신정호님 얼굴이 거꾸로 들어가있음(temp/faulty_frame.jpg) 이 옵션을 수정해서 정면 얼굴로 수정 시도해봄

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')
parser.add_argument('--no_segmentation', default=False, action='store_true',
					help='Prevent using face segmentation')
parser.add_argument('--no_sr', default=False, action='store_true',
					help='Prevent using super resolution')

parser.add_argument('--save_frames', default=False, action='store_true',
					help='Save each frame as an image. Use with caution')
parser.add_argument('--gt_path', type=str, 
					help='Where to store saved ground truth frames', required=False)
parser.add_argument('--pred_path', type=str, 
					help='Where to store frames produced by algorithm', required=False)
parser.add_argument('--save_as_video', action="store_true", default=False,
					help='Whether to save frames as video', required=False)
parser.add_argument('--image_prefix', type=str, default="",
					help='Prefix to save frames with', required=False)

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True


#### sr결과

class FaceEnhancement(object):
    def __init__(self, base_dir='./', in_size=512, out_size=512, model=None, use_sr=True, sr_model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, device=device)
        self.srmodel =  RealESRNet(base_dir, sr_model, device=device)
        self.faceparser = FaceParse(base_dir, device=device)
        self.use_sr = use_sr
        self.in_size = in_size
        self.out_size = out_size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.in_size, self.in_size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=20):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        return mask.astype(np.float32)

    def process(self, img, aligned=False):
        orig_faces, enhanced_faces = [], []
        if aligned:
            ef = self.facegan.process(img)
            orig_faces.append(img)
            enhanced_faces.append(ef)

            if self.use_sr:
                ef = self.srmodel.process(ef)

            return ef, orig_faces, enhanced_faces

        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])

        facebs, landms = self.facedetector.detect(img)
        
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.in_size, self.in_size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            #tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0]/255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            if self.in_size!=self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
        else:
            img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces



def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	"""
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	"""

	reader = read_frames()

	for i, m in enumerate(mels):
		try:
			frame_to_save = next(reader)
		except StopIteration:
			reader = read_frames()
			frame_to_save = next(reader)

		face, coords = face_detect([frame_to_save])[0]

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def read_frames():
	if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		face = cv2.imread(args.face)
		while 1:
			yield face

	video_stream = cv2.VideoCapture(args.face)
	fps = video_stream.get(cv2.CAP_PROP_FPS)

	print('Reading video frames from start...')

	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		if args.resize_factor > 1:
			frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

		if args.rotate:
			frame = cv2.rotate(frame, cv2.ROTATE_180)

		y1, y2, x1, x2 = args.crop
		if x2 == -1: x2 = frame.shape[1]
		if y2 == -1: y2 = frame.shape[0]

		frame = frame[y1:y2, x1:x2]

		yield frame

def main():
	frame_w = 0
	frame_h = 0

	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		fps = args.fps
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		frame_h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		print(f'fps : {fps}, frame_h : {frame_h}, frame_w : {frame_w}')
		video_stream.release()


	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	batch_size = args.wav2lip_batch_size
	gen = datagen(mel_chunks)



	if args.save_as_video:
		gt_out = cv2.VideoWriter("temp/gt.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (384, 384))
		pred_out = cv2.VideoWriter("temp/pred.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (96, 96))

	abs_idx = 0

	print("Loading segmentation network...")
	seg_net = init_parser(args.segmentation_path)

	print("Loading super resolution model...")
	#sr_net = init_sr_model(args.sr_path)
	face_enhancement = FaceEnhancement(base_dir='./', in_size=512, out_size=512, model='GPEN-BFR-512', use_sr=True, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, key=None, device='cuda')
	# face_enhacement 부분이 새로운 sr 방법 GPEN 방법으로 sr 모델을 여기서 불러옵니다!!
	# Model weight는  ./weight/ 폴더에 있습니다.
	model = load_model(args.checkpoint_path)
	print ("Model loaded")
######################## GPEN(SR) 적용시 반드시 frame_w,h 사이즈 2배 해줘야함
	output_resolution = (frame_w, frame_h)
	if not args.no_sr:
		output_resolution = (frame_w*2, frame_h*2)
	out = cv2.VideoWriter('temp/result.avi', 
							cv2.VideoWriter_fourcc(*'DIVX'), fps, output_resolution)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		""" if i == 0:
			print("Loading segmentation network...")
			seg_net = init_parser(args.segmentation_path)

			print("Loading super resolution model...")
			#sr_net = init_sr_model(args.sr_path)
			face_enhancement = FaceEnhancement(base_dir='./', in_size=512, out_size=512, model='GPEN-BFR-512', use_sr=True, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, key=None, device='cuda')
			# face_enhacement 부분이 새로운 sr 방법 GPEN 방법으로 sr 모델을 여기서 불러옵니다!!
			# Model weight는  ./weight/ 폴더에 있습니다.
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = next(read_frames()).shape[:-1]
			print('frame_h : {frame_h}, frame_w : {frame_w}')
######################## GPEN(SR) 적용시 반드시 frame_w,h 사이즈 2배 해줘야함
			out = cv2.VideoWriter('temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w*2, frame_h*2)) """

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			print("Generating lip...")
			pred = model(mel_batch, img_batch)

		print("Generated lip...")
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		print("Enhancing lip...")
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c

			if args.save_frames:
				if args.save_as_video:
					pred_out.write(p.astype(np.uint8))
					gt_out.write(cv2.resize(f[y1:y2, x1:x2], (384, 384)))
				else:
					cv2.imwrite(f"{args.gt_path}/{args.image_prefix}{abs_idx}.png", f[y1:y2, x1:x2])
					cv2.imwrite(f"{args.pred_path}/{args.image_prefix}{abs_idx}.png", p)
					abs_idx += 1

			#if not args.no_sr:
			#	p = enhance(sr_net, p)
###원래는 p = cenhace(sr_net, p)로 esrgan을 사용하지만 얼굴이 노란색으로 나오는 문제가 존재하기 때문에 GPEN으로 변경
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1)) ## wav2lip 결과를 다시 원본 얼굴 사이즈로 리사이즈
			
			if not args.no_segmentation:
				p = swap_regions(f[y1:y2, x1:x2], p, seg_net) ## wav2lip 결과에서 얼굴 영역 segementation해서 mask artifact 없애기 (얼굴영역 아닌곳은 원본 영상 들어가며 화질 유지)

			f[y1:y2, x1:x2] = p
			if not args.no_sr:
				f,_,_ = face_enhancement.process(f) ## GPEN 적용으로 얼굴 개선과 SR  
			out.write(f)
		print("Enhanced lip...")

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

	if args.save_frames and args.save_as_video:
		gt_out.release()
		pred_out.release()

		command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/gt.avi', args.gt_path)
		subprocess.call(command, shell=platform.system() != 'Windows')

		command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/pred.avi', args.pred_path)
		subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
	main()
