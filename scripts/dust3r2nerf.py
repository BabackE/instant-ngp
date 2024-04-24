from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import torch
import numpy as np
import lovely_tensors as lt
import math
import argparse
import json

OUT_PATH = "transfoms.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Run dust3r inference on a set of images and export a transforms.json file.")

    parser.add_argument("--weights", type=str, required=True, help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="schedule for global alignment")
    parser.add_argument("--iterations", type=int, default=300, help="Number of iterations")
    parser.add_argument("--input", type=str, required=True, help="directory to process")
    parser.add_argument("--rescale", type=int, default=0, choices=[0,1], help="rescale to original image size?")
    parser.add_argument("--invert", type=int, default=1, choices=[0,1], help="invert matrices for colmap?")
    parser.add_argument("--aabb_scale", default=128, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
    parser.add_argument("--nerf_adjustments", type=int, default=1, choices=[0, 1], help="transform and scale to nerf size?")

    args = parser.parse_args()
    return args

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

import cv2  # Assuming OpenCV is used for image saving
from PIL import Image

# creates directories colmap_scratch and images/dust3r_images and images/dust3r_masks
def init_filestructure(input_path, save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    images_path = Path(input_path) / "dust3r_images"
    masks_path = Path(input_path) / "dust3r_masks"
    sparse_path = save_path 
    
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)    
    sparse_path.mkdir(exist_ok=True, parents=True)
    
    return save_path, images_path, masks_path, sparse_path

# saves scaled down images in images/dust3r_images and masks in images/dust3r_masks
def save_images_masks(imgs, masks, images_path, masks_path):
    # Saving images and optionally masks/depth maps
    for i, (image, mask) in enumerate(zip(imgs, masks)):
        image_save_path = images_path / f"{i}.png"
        
        mask_save_path = masks_path / f"{i}.png"
        rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)        
        
# creates colmap_text/cameras.txt
def save_cameras(focals, principal_points, sparse_path, imgs_shape, image_files, rescale=False):
    # Save cameras.txt
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            if (not rescale):
                cameras_file.write(f"{i} SIMPLE_PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {pp[0]} {pp[1]}\n")
            else:
                # get the original image and calculate the scale factor from dust3r's pass
                this_source_image = Image.open(image_files[i])
                source_width, source_height = this_source_image.size
                x_scale_factor = float(source_width) / float(imgs_shape[2])
                y_scale_factor = float(source_height) / float(imgs_shape[1])
                
                cameras_file.write(f"{i} SIMPLE_PINHOLE {imgs_shape[2] * x_scale_factor} {imgs_shape[1] * y_scale_factor} {focal[0] * x_scale_factor} {pp[0] * x_scale_factor} {pp[1] * y_scale_factor}\n")
            
# creates colmap_text/images.txt
def save_imagestxt(cams2world, sparse_path, input_files, rescale=False):
     # Save images.txt
    images_file = sparse_path / 'images.txt'
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose_c2w in enumerate(cams2world):

            print (f"adding cam {i} {pose_c2w}")
            print (f"  rotation = {pose_c2w[ :3, :3]}")
            print (f"  trans = {pose_c2w[ :3, 3]}")
            print (f"  quat ={rotmat2qvec(pose_c2w[ :3, :3])}")            

            # Convert rotation matrix to quaternion
            out_name = f"dust3r_images/{i}.png" if (not rescale) else Path(input_files[i]).name
            rotation_matrix = pose_c2w[:3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = pose_c2w[:3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {out_name}\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

# creates transforms.json adapted from instant-ngp/scripts/colmap2nerf.py
def save_transforms(focals, principal_points, imgs_shape, cams2world, input_files, input_dir, AABB_SCALE=32, rescale=False, do_nerf_adj=True):
    print ("creating transforms.json")
    cameras = {}
    for i, (focal, pp) in enumerate(zip(focals, principal_points)):
        x_scale_factor = 1.0
        y_scale_factor = 1.0
        if (rescale):
            this_source_image = Image.open(image_files[i])
            source_width, source_height = this_source_image.size
            x_scale_factor = float(source_width) / float(imgs_shape[2])
            y_scale_factor = float(source_height) / float(imgs_shape[1])

        # pinhole camera parameters for all cameras
        camera = {}
        camera_id = i
        camera["w"] = imgs_shape[2] * x_scale_factor
        camera["h"] = imgs_shape[1] * y_scale_factor
        camera["fl_x"] = focal[0] * x_scale_factor
        camera["fl_y"] = focal[0] * y_scale_factor
        camera["k1"] = 0
        camera["k2"] = 0
        camera["k3"] = 0
        camera["k4"] = 0
        camera["p1"] = 0
        camera["p2"] = 0
        camera["cx"] = pp[0] * x_scale_factor
        camera["cy"] = pp[1] * y_scale_factor
        camera["is_fisheye"] = False
 
         # fl = 0.5 * w / tan(0.5 * angle_x);
        camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
        camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
        camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
        camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

        print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
        cameras[camera_id] = camera
    
    out = {
        "frames": [],
        "aabb_scale": AABB_SCALE,
        "n_extra_learnable_dims": 16,
    }
    up = np.zeros(3)
    for i, pose_c2w in enumerate(cams2world):
        name = str(Path(input_dir) / f"dust3r_images/{i}.png" if (not rescale) else Path(input_files[i]).name)
        b = sharpness(name)
        print(name, "sharpness=",b)
        c2w = pose_c2w  # export as is

        # reorientate for nerf friendliness
        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down        
        up += c2w[0:3,1] # accumulate the up

        frame = {"file_path":name,"sharpness":b,"transform_matrix": c2w}
        frame.update(cameras[i])
        out["frames"].append(frame)

    # now that all the c2ws are in place take the normalized up vector of all the cameras
    # and rotate them to the nerf's up vector so the interactive camera controls work
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    nframes = len(out["frames"])
    if do_nerf_adj:
		# find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"         

    for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

    print(nframes,"frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

if __name__ == '__main__':    
    args = parse_args()

    # get the image files
    from pathlib import Path
    Path.ls = lambda x: list(x.iterdir())
    image_dir = Path(args.input)
    image_files = [str(x) for x in image_dir.ls() if x.suffix in ['.png', '.jpg']]

    # do dust3r inference
    batch_size = 1
    lr = 0.01

    model = load_model(args.weights, args.device)
    images = load_images(image_files, size=int(args.image_size))
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=batch_size)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=args.iterations, schedule=args.schedule, lr=lr)

    # get the camera and scene parameters
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    cams2world = scene.get_im_poses().detach().cpu().numpy()
    if (bool(args.invert)):
        print("inverting cams2world")
        cams2world = inv(cams2world)
    else:
        print("keeping cams2world in dust3r coords")

    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)
    #pts3d = [i.detach() for i in scene.get_pts3d()]
    #depth_maps = [i.detach() for i in scene.get_depthmaps()]

    min_conf_thr = 20
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())

    # save to colmap_text
    save_dir = Path("colmap_text")
    save_dir.mkdir(exist_ok=True, parents=True)

    save_path, images_path, masks_path, sparse_path = init_filestructure(args.input, save_dir)
    save_images_masks(imgs, masks, images_path, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape, image_files=image_files, rescale=bool(args.rescale))
    save_imagestxt(cams2world, sparse_path, input_files=image_files, rescale=bool(args.rescale))
    save_transforms(focals, principal_points, imgs.shape, cams2world, image_files, image_dir, args.aabb_scale, args.rescale, args.nerf_adjustments)

