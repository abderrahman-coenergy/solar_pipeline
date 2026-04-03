import numpy as np
from typing import Tuple, Optional, Union
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image
from pathlib import Path


class Equirectangular:
    """Contains an equirectangular projection of the sphere (theta x phi)."""
    def __init__(self, array=None, theta=None, phi=None):
        if array is None:
            array = np.zeros((180, 360, 3), dtype=np.uint8)
        self.array = np.asarray(array).astype(np.uint8)
        self.h, self.w = self.array.shape[:2]
        self.rotation = (0.0, 0.0, 0.0)
        if theta is None:
            self.theta = (np.arange(self.h) + 0.5) / self.h * np.pi
        else:
            self.theta = np.asarray(theta)
        if phi is None:
            self.phi = (np.arange(self.w) + 0.5) / self.w * 2.0 * np.pi - np.pi
        else:
            self.phi = np.asarray(phi)

    def to_pil(self):
        return Image.fromarray(self.array)

    def to_numpy(self):
        return self.array

    def flipped(self):
        return Equirectangular(np.flipud(self.array), theta=np.flipud(self.theta), phi=self.phi)

    def luminance(self):
        rgb = self.array.astype(np.float32) / 255.0
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])

    def preview(self, show_2d=True, show_3d=True, figsize=(10, 4)):
        if show_2d:
            import matplotlib.pyplot as plt
            plt.figure(figsize=figsize)
            plt.imshow(self.to_numpy())
            plt.axis('off')
        if show_3d:
            try:
                import plotly.graph_objects as go
                import plotly.io as pio
            except Exception:
                return
            eq = self.flipped()
            oh, ow = eq.h, eq.w
            theta = eq.theta
            phi = eq.phi
            Phi, Theta = np.meshgrid(phi, theta)
            X = np.sin(Theta) * np.cos(Phi)
            Y = np.sin(Theta) * np.sin(Phi)
            Z = np.cos(Theta)
            verts_x = X.flatten()
            verts_y = Y.flatten()
            verts_z = Z.flatten()
            I = []
            J = []
            K = []
            for r in range(oh - 1):
                for c in range(ow):
                    c_next = (c + 1) % ow
                    v0 = r * ow + c
                    v1 = r * ow + c_next
                    v2 = (r + 1) * ow + c
                    v3 = (r + 1) * ow + c_next
                    I.append(v0); J.append(v2); K.append(v1)
                    I.append(v1); J.append(v2); K.append(v3)
            rgb = eq.array.astype(np.uint8).reshape(-1, 3)
            cols = [f'rgb({r},{g},{b})' for (r, g, b) in rgb]
            mesh = go.Mesh3d(x=verts_x, y=verts_y, z=verts_z, i=I, j=J, k=K, vertexcolor=cols, flatshading=False)
            fig = go.Figure(data=[mesh])
            fig.update_layout(title='Preview: spherical (RGB)', scene=dict(aspectmode='data'))
            try:
                fig.show()
            except Exception:
                pio.renderers.default = 'browser'
                fig.show()

    def rotate(self, delta_azimuth: float = 0.0, delta_inclination: float = 0.0, delta_roll: float = 0.0):
        eps = 1e-12
        if abs(float(delta_azimuth)) < eps and abs(float(delta_inclination)) < eps and abs(float(delta_roll)) < eps:
            return

        h, w = self.h, self.w
        yaw = float(delta_azimuth)
        pitch = float(delta_inclination)
        roll = float(delta_roll)
        R_cam2world = self._compute_R_cam2world(yaw, pitch, roll)

        if abs(pitch) < eps and abs(roll) < eps:
            col_shift = int(round((yaw / (2.0 * np.pi)) * float(w)))
            if col_shift != 0:
                self.array = np.roll(self.array, shift=col_shift, axis=1)
            ra, ri, rr = self.rotation
            self.rotation = (ra + yaw, ri + pitch, rr + roll)
            return

        phi_grid, theta_grid, Vx, Vy, Vz = self._spherical_dirs()
        V = np.stack([Vx, Vy, Vz], axis=-1).reshape(-1, 3)
        V_src = np.dot(V, R_cam2world.T)
        x_s = V_src[:, 0].reshape(Vx.shape)
        y_s = V_src[:, 1].reshape(Vy.shape)
        z_s = V_src[:, 2].reshape(Vz.shape)
        z_s = np.clip(z_s, -1.0, 1.0)
        theta_src = np.arccos(z_s)
        phi_src = np.arctan2(y_s, x_s)
        row_f = (theta_src / np.pi) * float(h) - 0.5
        col_f = ((phi_src + np.pi) / (2.0 * np.pi)) * float(w) - 0.5
        map_x_cv = (col_f % float(w)).astype(np.float32)
        map_y_cv = np.clip(row_f, 0.0, float(h) - 1.0).astype(np.float32)
        arr = self._apply_remap(map_x_cv, map_y_cv)
        self.array = arr
        ra, ri, rr = self.rotation
        self.rotation = (ra + yaw, ri + pitch, rr + roll)

    def _compute_R_cam2world(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Return camera-to-world rotation matrix for given yaw,pitch,roll (radians)."""
        cyaw = np.cos(yaw); syaw = np.sin(yaw)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cr = np.cos(roll); sr = np.sin(roll)
        R_yaw = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]])
        R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        R_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return R_yaw.dot(R_pitch).dot(R_roll)

    def _spherical_dirs(self):
        """Return phi_grid,theta_grid and unit vectors Vx,Vy,Vz for this map."""
        phi_grid, theta_grid = np.meshgrid(self.phi, self.theta)
        s = np.sin(theta_grid)
        Vx = s * np.cos(phi_grid)
        Vy = s * np.sin(phi_grid)
        Vz = np.cos(theta_grid)
        return phi_grid, theta_grid, Vx, Vy, Vz

    def _apply_remap(self, map_x_cv: np.ndarray, map_y_cv: np.ndarray) -> np.ndarray:
        """Remap self.array according to float maps; use OpenCV if available."""
        if cv2 is not None:
            try:
                interp = cv2.INTER_LANCZOS4
            except Exception:
                interp = cv2.INTER_CUBIC
            remapped = cv2.remap(self.array, map_x_cv, map_y_cv, interpolation=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return remapped.astype(np.uint8)
        else:
            h, w = self.h, self.w
            rf = map_y_cv
            cf = map_x_cv
            r0 = np.floor(rf).astype(int)
            c0 = np.floor(cf).astype(int) % w
            r1 = np.clip(r0 + 1, 0, h - 1)
            c1 = (c0 + 1) % w
            wr = rf - r0
            wc = cf - c0
            if self.array.ndim == 3:
                out = np.empty_like(self.array)
                for ch in range(self.array.shape[2]):
                    v00 = self.array[r0, c0, ch]
                    v10 = self.array[r1, c0, ch]
                    v01 = self.array[r0, c1, ch]
                    v11 = self.array[r1, c1, ch]
                    val = (1 - wr) * ((1 - wc) * v00 + wc * v01) + wr * ((1 - wc) * v10 + wc * v11)
                    out[..., ch] = val
            else:
                v00 = self.array[r0, c0]
                v10 = self.array[r1, c0]
                v01 = self.array[r0, c1]
                v11 = self.array[r1, c1]
                out = (1 - wr) * ((1 - wc) * v00 + wc * v01) + wr * ((1 - wc) * v10 + wc * v11)
            return np.clip(out, 0, 255).astype(np.uint8)

    def add_image(self, img: Union[np.ndarray, Image.Image, str, Path], pic_vfov_deg: float, pic_hfoc_deg: float, pic_pitch: float, pic_yaw: float, pic_roll: float):
        """High-level wrapper: project `img` into this equirectangular map and overlay.

        This method delegates work to smaller helpers for clarity and testability.
        """
        img_np = self._load_rgb_image(img)
        ih, iw = img_np.shape[:2]
        fx, fy, cx, cy = self._compute_focal_pixels(ih, iw, pic_vfov_deg, pic_hfoc_deg)
        map_x, map_y = self._build_world_to_image_maps(fx, fy, cx, cy, pic_pitch, pic_yaw, pic_roll)
        sampled = _sample_image(img_np, map_x, map_y, cv2)
        self._overlay_sampled(sampled, map_x)

    def _load_rgb_image(self, img: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Load/convert input to an HxWx3 uint8 RGB numpy array."""
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert('RGB')
            return np.array(img).astype(np.uint8)
        if isinstance(img, Image.Image):
            return np.array(img.convert('RGB')).astype(np.uint8)
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr.astype(np.uint8)

    def _compute_focal_pixels(self, ih: int, iw: int, vfov_deg: float, hfov_deg: float) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) for a camera with given image size and FOVs."""
        vfov_rad = np.deg2rad(float(vfov_deg))
        hfov_rad = np.deg2rad(float(hfov_deg))
        fy = (float(ih) / 2.0) / np.tan(vfov_rad / 2.0)
        fx = (float(iw) / 2.0) / np.tan(hfov_rad / 2.0)
        cx = float(iw) / 2.0
        cy = float(ih) / 2.0
        return fx, fy, cx, cy

    def _build_world_to_image_maps(self, fx: float, fy: float, cx: float, cy: float, pic_pitch: float, pic_yaw: float, pic_roll: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-equirectangular-pixel image coordinates (map_x, map_y).

        Returns maps with -1 for pixels that do not project into the picture.
        """
        # spherical directions at each equirectangular pixel
        phi_grid, theta_grid = np.meshgrid(self.phi, self.theta)
        Vx, Vy, Vz = _unit_vectors(theta_grid, phi_grid)
        Hout, Wout = Vx.shape
        V = np.stack([Vx, Vy, Vz], axis=-1).reshape(-1, 3)

        # build rotation (camera -> world) then invert
        yaw = np.deg2rad(float(pic_yaw))
        pitch = np.deg2rad(float(pic_pitch))
        roll = np.deg2rad(float(pic_roll))
        cyaw = np.cos(yaw); syaw = np.sin(yaw)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cr = np.cos(roll); sr = np.sin(roll)
        R_yaw = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]])
        R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        R_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        R_cam2world = R_yaw.dot(R_pitch).dot(R_roll)
        R_world2cam = R_cam2world.T

        V_cam = np.dot(V, R_world2cam.T)
        Xc = V_cam[:, 0]
        Yc = V_cam[:, 1]
        Zc = V_cam[:, 2]

        valid = Xc > 1e-9
        px = np.full_like(Xc, -1.0, dtype=np.float32)
        py = np.full_like(Xc, -1.0, dtype=np.float32)
        px[valid] = cx + fx * (Yc[valid] / Xc[valid])
        py[valid] = cy - fy * (Zc[valid] / Xc[valid])

        map_x = px.reshape(Hout, Wout).astype(np.float32)
        map_y = py.reshape(Hout, Wout).astype(np.float32)
        # keep out-of-image values negative to indicate invalid
        return map_x, map_y

    def _overlay_sampled(self, sampled: np.ndarray, map_x: np.ndarray):
        """Overlay sampled image onto self.array where sampled is non-black and mapping valid."""
        mask_valid_map = (map_x != -1.0)
        if sampled.ndim == 3:
            mask_nonblack = (sampled[..., 0] != 0) | (sampled[..., 1] != 0) | (sampled[..., 2] != 0)
        else:
            mask_nonblack = sampled != 0
        mask = mask_valid_map & mask_nonblack
        if self.array.ndim == 3:
            self.array[mask] = sampled[mask]
        else:
            lum = (0.299 * sampled[..., 0] + 0.587 * sampled[..., 1] + 0.114 * sampled[..., 2]).astype(self.array.dtype)
            self.array[mask] = lum[mask]



def _prepare_input(img: Union[np.ndarray, Image.Image, str, Path], single_half: Optional[str]) -> Tuple[np.ndarray, int, int, int]:
    """Convert input to numpy array and optionally zero the missing half.

    Args:
        img: Input image as a numpy array or PIL Image. Expected HxW(x3).
        single_half: If 'left' or 'right', the opposite half will be set to black.

    Returns:
        Tuple of (img_numpy, height, width, half_width).
    """
    if isinstance(img, (str, Path)):
        img = Image.open(str(img)).convert('RGB')
        img = np.array(img)
    elif isinstance(img, Image.Image):
        img = np.array(img.convert('RGB'))
    img = np.asarray(img)
    h, W = img.shape[:2]
    w_half = W // 2
    if single_half is not None:
        if single_half not in ('left', 'right'):
            raise ValueError("single_half must be one of: None, 'left', 'right'")
        img = img.copy()
        if single_half == 'left':
            img[:, w_half:] = 0
        else:  # 'right'
            img[:, :w_half] = 0
    return img, h, W, w_half


def _compute_centers_and_f(h: int, w_half: int, fov_deg: float) -> Tuple[float, float, float, float, float, float]:
    """Compute image centers and focal parameter for equidistant fisheye model.

    Returns: (cx_left, cx_right, cy, R, fov_rad, f)
    """
    cx_left = float(w_half) / 2.0
    cx_right = float(w_half) + cx_left
    cy = float(h) / 2.0
    R = min(cx_left, cy)
    fov_rad = np.deg2rad(fov_deg)
    f = R / (fov_rad / 2.0)
    return cx_left, cx_right, cy, R, fov_rad, f


def _spherical_grids(out_h: int, out_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build theta/phi sampling arrays and their meshgrids.

    - theta in [0, pi] with length out_h
    - phi in [-pi, pi] with length out_w
    Returns: theta, phi, phi_grid, theta_grid
    """
    theta = (np.arange(out_h) + 0.5) / out_h * np.pi
    phi = (np.arange(out_w) + 0.5) / out_w * 2.0 * np.pi - np.pi
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    return theta, phi, phi_grid, theta_grid


def _unit_vectors(theta_grid: np.ndarray, phi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit sphere direction vectors from theta/phi grids.

    Returns (Vx, Vy, Vz) arrays matching the input grid shape.
    """
    s = np.sin(theta_grid)
    Vx = s * np.cos(phi_grid)
    Vy = s * np.sin(phi_grid)
    Vz = np.cos(theta_grid)
    return Vx, Vy, Vz


def _build_maps(Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray, f: float, cx_left: float, cx_right: float, cy: float, fov_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build mapping arrays (map_x,map_y) mapping sphere directions to image coordinates.

    - Vx,Vy,Vz: unit direction components on the sphere
    - f: focal parameter (equidistant model)
    - cx_left/cx_right/cy: fisheye centers in image coordinates
    - fov_rad: fisheye full field of view in radians

    Returns: (map_x, map_y) arrays of same shape as Vx (float32). Points outside FOV have value -1.
    """
    out_h, out_w = Vx.shape
    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    mask_right = Vz >= 0
    if np.any(mask_right):
        Xc = Vy[mask_right]
        Yc = -Vx[mask_right]
        Zc = Vz[mask_right]
        Zc = np.clip(Zc, -1.0, 1.0)
        theta_p = np.arccos(Zc)
        r = f * theta_p
        denom = np.sqrt(Xc * Xc + Yc * Yc)
        ux = Xc / (denom + 1e-12)
        uy = Yc / (denom + 1e-12)
        x_img = cx_right + r * ux
        y_img = cy + r * uy
        visible = theta_p <= (fov_rad / 2.0)
        map_x[mask_right] = np.where(visible, x_img, -1.0)
        map_y[mask_right] = np.where(visible, y_img, -1.0)

    mask_left = ~mask_right
    if np.any(mask_left):
        Xc = Vy[mask_left]
        Yc = -Vx[mask_left]
        Zc = -Vz[mask_left]
        Zc = np.clip(Zc, -1.0, 1.0)
        theta_p = np.arccos(Zc)
        r = f * theta_p
        denom = np.sqrt(Xc * Xc + Yc * Yc)
        ux = Xc / (denom + 1e-12)
        uy = Yc / (denom + 1e-12)
        x_img = cx_left + r * ux
        y_img = cy + r * uy
        visible = theta_p <= (fov_rad / 2.0)
        map_x[mask_left] = np.where(visible, x_img, -1.0)
        map_y[mask_left] = np.where(visible, y_img, -1.0)

    return map_x, map_y


def _sample_image(img, map_x, map_y, cv2mod):
    out_h, out_w = map_x.shape
    if cv2mod is not None:
        map_x_cv = map_x.astype(np.float32)
        map_y_cv = map_y.astype(np.float32)
        return cv2mod.remap(img.astype(np.uint8), map_x_cv, map_y_cv, interpolation=cv2mod.INTER_LINEAR, borderMode=cv2mod.BORDER_CONSTANT, borderValue=0)
    else:
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        xs = np.round(map_x).astype(int)
        ys = np.round(map_y).astype(int)
        valid = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        out[valid] = img[ys[valid], xs[valid]]
        return out

def dual_fisheye_to_equirectangular(img, out_h=180, out_w=360, fov_deg=180, single_half=None):
    """
    Convert a dual-fisheye image (two fisheyes side-by-side) into an Equirectangular object.
    Args:
      img: HxWx3 numpy array (RGB) or PIL Image. Assumes the two fisheye images are left and right halves.
      out_h,out_w: output map size (lat x lon) - default 180x360 (1 deg per pixel approx).
      fov_deg: field-of-view (full angle) of a single fisheye in degrees (e.g. 180).
      If single_half is provided (either 'left' or 'right'), the other half is assumed to be pitch black.

    Assumptions:
        We assume that the fisheye follows the equidistant projection model:
        r = f * theta, with f = R / (FOV_rad/2), where R is the radius of the fisheye circle in pixels, and theta is the angle from the optical axis.
    Returns:
      `Equirectangular` instance wrapping the produced image.
    """
    # Robust, self-contained vectorized implementation.
    img, h, W, w_half = _prepare_input(img, single_half)
    # Fisheye centers and focal parameter (equidistant model)
    cx_left = float(w_half) / 2.0
    cx_right = float(w_half) + cx_left
    cy = float(h) / 2.0
    R = min(cx_left, cy)
    fov_rad = np.deg2rad(float(fov_deg))
    f = R / (fov_rad / 2.0)

    # build spherical sampling grids
    theta = (np.arange(out_h) + 0.5) / out_h * np.pi
    phi = (np.arange(out_w) + 0.5) / out_w * 2.0 * np.pi - np.pi
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # unit vectors for each equirectangular pixel
    s = np.sin(theta_grid)
    Vx = s * np.cos(phi_grid)
    Vy = s * np.sin(phi_grid)
    Vz = np.cos(theta_grid)

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    eps = 1e-12
    # Simpler hemisphere-based mapping:
    # - upper hemisphere (theta < pi/2) -> right fisheye, zenith at center
    # - lower hemisphere (theta >= pi/2) -> left fisheye, nadir at center
    # Use equidistant (linear) mapping: r = f * theta_p where theta_p is angle from fisheye optical axis.
    theta_grid_small = theta_grid
    phi_grid_small = phi_grid

    # unit direction in XY plane (phi orientation)
    ux = np.cos(phi_grid_small)
    uy = np.sin(phi_grid_small)

    # Upper hemisphere
    mask_upper = theta_grid_small <= (np.pi / 2.0)
    if np.any(mask_upper):
        theta_p = theta_grid_small[mask_upper]  # angle from zenith
        r_vals = f * theta_p
        x_img = cx_right + r_vals * ux[mask_upper]
        y_img = cy - r_vals * uy[mask_upper]
        visible = theta_p <= (fov_rad / 2.0)
        map_x[mask_upper] = np.where(visible, x_img, -1.0)
        map_y[mask_upper] = np.where(visible, y_img, -1.0)

    # Lower hemisphere
    mask_lower = ~mask_upper
    if np.any(mask_lower):
        theta_p = (np.pi - theta_grid_small[mask_lower])  # angle from nadir
        r_vals = f * theta_p
        x_img = cx_left + r_vals * ux[mask_lower]
        y_img = cy - r_vals * uy[mask_lower]
        visible = theta_p <= (fov_rad / 2.0)
        map_x[mask_lower] = np.where(visible, x_img, -1.0)
        map_y[mask_lower] = np.where(visible, y_img, -1.0)

    # Sample using OpenCV remap if available, else fallback to nearest sampling
    if cv2 is not None:
        map_x_cv = map_x.astype(np.float32)
        map_y_cv = map_y.astype(np.float32)
        out = cv2.remap(img.astype(np.uint8), map_x_cv, map_y_cv, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        xs = np.round(map_x).astype(int)
        ys = np.round(map_y).astype(int)
        valid = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        out[valid] = img[ys[valid], xs[valid]]

    return Equirectangular(out, theta=theta, phi=phi)

if __name__ == "__main__":

    img_path = '/Users/eliotjanvier/Documents/freelance/coe/cosol/frame_26s.jpg'
    eq = dual_fisheye_to_equirectangular(img_path, out_h=180, out_w=360, fov_deg=180, single_half=None)
    eq.preview(show_2d=True, show_3d=True)