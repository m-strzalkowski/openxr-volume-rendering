import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { ARButton } from 'three/addons/ARButton.js';
import { VRButton } from 'three/addons/VRButton.js';
import { HEADER, loadVolumeData } from './volume.js';
import { T, LANG, OTHER_LANG, setLang, localiseXrButton } from './i18n.js';

// ---------------------------------------------------------------- parameters

const qs = new URLSearchParams(location.search);
const num = (k, d) => (qs.has(k) ? Number(qs.get(k)) : d);

const SQRT3 = Math.sqrt(3);
const NB = 80;                      // voxels per side

// Sampling below ~1 sample/voxel along the box diagonal (80*sqrt3 ~ 139 steps)
// beats against the voxel grid and shows up as slices. Trilinear filtering
// means full Nyquist (277) is not needed; ~1.5 samples/voxel is where it stops
// being visible, which is the 200 found by experiment.
const STEPS_DEFAULT = Math.ceil(1.5 * NB * SQRT3);   // 208

// How the coarse sensor map is reconstructed when magnified ~5x to screen.
// 1 = cubic B-spline, 0 = the older smoothstep-weighted bilinear. The B-spline
// needs no averaging on top -- its smoothness comes from the filter rather than
// from blurring afterwards -- so the tap count defaults differently for each.
const ENV_FILTER = num('envfilter', 1);

const P = {
	dvc: num('dvc', 503),
	limit: num('limit', 0),
	steps: num('steps', STEPS_DEFAULT),
	gain: num('gain', 60),
	threshold: num('threshold', 0),
	size: num('size', 0.8),        // cube edge in metres
	dist: num('dist', 1.2),        // metres in front of the viewer
	height: num('height', 1.3),
	fbscale: num('fbscale', 1.0),  // XR framebuffer scale (applied at session start)
	rtscale: num('rtscale', 1.0),  // starting volume-buffer scale
	adapt: num('adapt', 1),        // 0 pins the resolution
	rt: num('rt', 1),              // 0 draws the volume straight to the screen
	debug: num('debug', 0),        // 1 logs XR state once a second
	depth: num('depth', 1),        // 0 stops us requesting depth-sensing at all
	depthnear: num('depthnear', 0), // >0 overrides the sensor near plane, for calibrating
	envfilter: ENV_FILTER,         // 1 = cubic B-spline, 0 = smoothstep bilinear
	envtaps: num('envtaps', ENV_FILTER === 1 ? 1 : 5),  // 9 (3x3), 5 (quincunx) or 1
	envscale: num('envscale', 0.53),// prepass buffer size, relative to the framebuffer
	envsoft: num('envsoft', 1.2),  // prepass kernel spread, in sensor texels
	volocc: num('volocc', 1),      // 0 leaves the cloud un-occluded by the room
	ruler: num('ruler', 0),        // 1 = distance readout at the centre of the view
	occlo: num('occlo', 0.6),    // coverage below this reads as no occlusion
	occhi: num('occhi', 0.8),        // coverage at or above this reads as full
	depthrange: num('depthrange', 5),  // metres across the inspector's colour ramp
	occbias: num('occbias', 0.05), // banner occlusion: depth offset, metres
	occsoft: num('occsoft', 1.0),  // banner occlusion: tap spread, in sensor texels
	light: qs.get('light') === '1',
};

const status = document.getElementById('status');
const hud = document.getElementById('hud');
const say = (msg, err = false) => {
	status.textContent = msg;
	status.className = err ? 'err' : '';
};

document.documentElement.lang = LANG;
document.title = T.title;
say(T.loadingVolume);

// Flat page only: a reload is the whole mechanism, and there is nothing in a
// session that would survive one. Hidden while presenting, because the DOM
// overlay puts the whole body in front of the viewer.
const langBtn = document.getElementById('lang-btn');
langBtn.textContent = T.langOther;
langBtn.addEventListener('click', () => setLang(OTHER_LANG));

// ---------------------------------------------------------------- volume shader

const VOL_VERT = /* glsl */`
out vec3 vPos;
void main() {
	vPos = position + 0.5;            // unit cube [-0.5,0.5] -> texture coords
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// Reads only the prepass buffer, never the sensor texture, so this material is
// built once and never needs rebuilding when the depth format turns up.
function volumeFragment() {
	return /* glsl */`
precision highp float;
precision highp sampler3D;

uniform sampler2D envTex;     // prepass: r = metres, g = occlusion coverage
uniform vec4  envRect;        // this eye's rectangle inside envTex, in pixels
uniform vec2  envSize;        // envTex dimensions
uniform vec4  dstRect;        // this view's rectangle, for the screen lookup
uniform int   useDepth;

uniform sampler3D volume;
uniform vec3  bbMin;           // tight box around everything above threshold
uniform vec3  bbMax;
uniform vec3  camPos;          // camera in texture space, set per eye
uniform float densityGain;
uniform float threshold;
uniform float stepLen;         // fixed, in texture units
uniform int   maxSteps;
uniform int   lightMode;

in vec3 vPos;
out vec4 fragColor;

// A large finite number instead of Inf. For a ray exactly parallel to an axis,
// 1.0/0.0 is Inf, and if the matching origin component is also 0 the product is
// 0*Inf = NaN, which then poisons the min/max chain and surfaces as a stray
// white pixel. Keeping the reciprocal finite keeps the whole chain finite.
vec3 safeRcp(vec3 v) {
	vec3 s = vec3(v.x < 0.0 ? -1.0 : 1.0, v.y < 0.0 ? -1.0 : 1.0, v.z < 0.0 ? -1.0 : 1.0);
	return s / max(abs(v), vec3(1e-8));
}

vec2 boxIntersect(vec3 ro, vec3 rd, vec3 lo, vec3 hi) {
	vec3 inv = safeRcp(rd);
	vec3 t0 = (lo - ro) * inv;
	vec3 t1 = (hi - ro) * inv;
	vec3 tmn = min(t0, t1), tmx = max(t0, t1);
	return vec2(max(max(tmn.x, tmn.y), tmn.z),
	            min(min(tmx.x, tmx.y), tmx.z));
}

vec3 ramp(vec3 c1, vec3 c2, vec3 c3, vec3 c4, float t) {
	t = clamp(t, 0.0, 1.0);
	if (t < 0.33)      return mix(c1, c2, t / 0.33);
	else if (t < 0.66) return mix(c2, c3, (t - 0.33) / 0.33);
	else               return mix(c3, c4, (t - 0.66) / 0.34);
}

vec3 colormap(float t) {
	return (lightMode == 1)
		? ramp(vec3(0.70, 0.90, 0.97), vec3(0.15, 0.55, 0.95),
		       vec3(0.20, 0.10, 0.70), vec3(0.14, 0.03, 0.20), t)
		: ramp(vec3(0.10, 0.03, 0.35), vec3(0.75, 0.15, 0.30),
		       vec3(0.99, 0.65, 0.15), vec3(1.00, 0.98, 0.75), t);
}

void main() {
	// normalize() of a zero vector is NaN, reachable when the eye lands exactly
	// on the cube surface while walking through the wall.
	vec3 dir = vPos - camPos;
	float dlen = length(dir);
	if (dlen < 1e-7) discard;
	vec3 rd = dir / dlen;

	// Marched against the tight box, not the whole cube. Nothing outside it can
	// pass the threshold, so the image is unchanged -- but the interval is
	// shorter, so the same step count lands more samples on the part that
	// matters, and rays that miss the box entirely cost nothing.
	vec2 t = boxIntersect(camPos, rd, bbMin, bbMax);
	float tmin = max(t.x, 0.0);
	float tmax = t.y;
	if (tmax <= tmin) discard;

	// Environment occlusion. The ray starts at the camera, so the camera is the
	// view-space origin and eye depth is exactly linear in t:
	//     eyeZ(t) = -t * (MV * vec4(rd,0)).z = t * k
	// One dot product absorbs the group rotation, the metres-per-unit scale and
	// the camera transform, so occluding is a single clamp of tmax rather than
	// a test inside the loop -- and a shorter ray is cheaper, not dearer.
	if (tmax <= tmin) discard;

	// Fixed step length, not (tmax-tmin)/N. With a segment-proportional dt the
	// longest rays -- straight through the middle of the cloud, which is what
	// you actually look at -- get the coarsest sampling, while rays clipping a
	// corner are wildly oversampled. A constant dt makes quality uniform, makes
	// per-sample opacity independent of view geometry, costs less on short
	// rays, and stops the box geometry from being imprinted on the sampling.
	//
	// There is deliberately no dither/jitter of the ray start. An earlier
	// version hashed the entry point: inside the box that collapses to a single
	// constant (tmin is 0 for every pixel), and outside it renders the hash
	// function itself as moire fringes. With >=1 sample/voxel it buys nothing.
	float dt = stepLen;
	vec3  acc = vec3(0.0);
	float trans = 1.0;

	for (int i = 0; i < maxSteps; i++) {
		float t0 = tmin + float(i) * dt;
		if (t0 >= tmax) break;
		// The final segment is clipped to the exact stopping point -- the box
		// exit, or the real surface. Sampling its midpoint and attenuating over
		// its true length puts the cut where the depth says it is, instead of
		// quantising it to the last whole step.
		float seg = min(dt, tmax - t0);
		float d = texture(volume, camPos + rd * (t0 + seg * 0.5)).r;
		if (d > 0.001 && d > threshold) {
			float a = 1.0 - exp(-d * densityGain * seg);
			acc += trans * a * colormap(d);
			trans *= 1.0 - a;
			if (trans < 0.01) break;
		}
	}
	// One bilinear fetch of a coverage that was already antialiased in the
	// prepass. Nothing here thresholds anything, so nothing re-sharpens the
	// silhouette; full coverage hides the cloud completely, and the soft band
	// exists only where the prepass taps disagreed -- that is, at edges.
	float vis = 1.0;
	if (useDepth == 1) {
		vec2 viewUv = clamp((gl_FragCoord.xy - dstRect.xy) / dstRect.zw, 0.0, 1.0);
		vis = 1.0 - texture(envTex, (envRect.xy + viewUv * envRect.zw) / envSize).g;
	}
	fragColor = vec4(acc, 1.0 - trans) * vis;   // premultiplied
}
`;
}

// ---------------------------------------------------------------- composite shader

// Blits the low-resolution volume buffer over the full-resolution frame. Driven
// entirely by gl_FragCoord and two rectangles, so the same quad works for a
// plain canvas and for each eye of a stereo XR framebuffer.
const BLIT_VERT = /* glsl */`
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const BLIT_FRAG = /* glsl */`
precision highp float;
uniform sampler2D tex;
uniform vec4 dstRect;   // this view's rectangle in the destination framebuffer
uniform vec4 srcRect;   // the matching rectangle inside the volume buffer
uniform vec2 texSize;
out vec4 fragColor;
void main() {
	vec2 f = clamp((gl_FragCoord.xy - dstRect.xy) / dstRect.zw, 0.0, 1.0);
	// Map to texel CENTRES, from src+0.5 to src+size-0.5. A bilinear tap reaches
	// half a texel either side, and immediately outside this rectangle is the
	// other eye -- sampling the rect edge directly bleeds one eye into the other
	// along the seam, which against a black background reads as a white flicker.
	vec2 px = srcRect.xy + 0.5 + f * (srcRect.zw - 1.0);
	fragColor = texture(tex, px / texSize);
}
`;

// ---------------------------------------------------------------- depth view

// Quest 3's environment depth, seen through the WebXR Depth Sensing Module
// rather than the native XR_META_environment_depth extension.
//
// three's own occlusion shader tells us the shape of the data: the
// gpu-optimized texture is a sampler2DArray with ONE LAYER PER EYE, and three
// writes .r straight to gl_FragDepth -- so it is projected depth in [0,1], not
// metres. That is why this shader un-projects with the session's depthNear and
// depthFar before colouring: raw values all sit within a whisker of 1.0 and a
// plain greyscale of them looks blank.
// The two plausible readings of the sampled value are shown side by side,
// because which one applies depends on what the runtime handed us and guessing
// costs a trip to the headset each time:
//
//   left half  - raw * rawValueToMeters, the standard Depth Sensing Module
//   right half - un-projected with depthNear/depthFar, the "depth projection"
//                variant, where the value is NDC depth usable as gl_FragDepth
//
// Whichever half shows sane structure is the correct interpretation.
// How a sample becomes metres depends on the format the runtime chose, and the
// formats disagree completely:
//
//   luminance-alpha - a uint16 split across two channels. Reading .r alone gives
//                     the low byte only, which scaled by rawValueToMeters
//                     (~0.001) collapses everything to nearly zero. This is what
//                     made the headset show a near-white field with only a hand
//                     faintly visible, while the emulator -- handing back
//                     float32 -- looked fine.
//   float32         - .r is already the raw value
//   unsigned-short  - .r is normalised, so scale back up by 65535
function depthDecode(format) {
	switch (format) {
		case 'luminance-alpha':
			return 'dot(texel.ra, vec2(255.0, 255.0 * 256.0))';
		case 'unsigned-short':
			return 'texel.r * 65535.0';
		default:
			return 'texel.r';   // float32
	}
}

// Inspection view. Stage 0 shows the sensor exactly as it arrives -- nearest
// fetch, no interpolation -- so the true resolution is visible; the later
// stages show what each processing step made of it. Turbo rather than the
// volume ramp, because that one saturates to cream and hides the near field.
const DEPTH_STAGES = T.depthStages;

function depthFragment(isArray, format) {
	return /* glsl */`
precision highp float;
${isArray ? 'precision highp sampler2DArray;\nuniform sampler2DArray depthTex;'
			: 'uniform sampler2D depthTex;'}
uniform sampler2D envTex;
uniform vec4  envRect;
uniform vec2  envSize;
uniform vec4  dstRect;
uniform vec2  zbParams;   // Meta's _EnvironmentDepthZBufferParams (x, y)
uniform mat4  depthUvXform;
uniform float range;      // metres mapped across the colour ramp
uniform float rawScale;   // rawValueToMeters
uniform int   layer;
uniform int   stage;
out vec4 fragColor;

// Turbo, the usual polynomial fit. Monotonic in lightness and it uses its whole
// range, so near distances stay readable instead of clipping to white.
vec3 turbo(float t) {
	t = clamp(t, 0.0, 1.0);
	return clamp(vec3(
		34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))),
		23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56)))),
		27.20 + t * (3211.10 + t * (-15327.97 + t * (27814.00 + t * (-22569.18 + t * 6838.66))))
	) / 255.0, 0.0, 1.0);
}

void main() {
	vec2 viewUv = clamp((gl_FragCoord.xy - dstRect.xy) / dstRect.zw, 0.0, 1.0);
	float metres;

	if (stage == 0) {
		vec2 uv = (depthUvXform * vec4(viewUv, 0.0, 1.0)).xy;
		vec4 texel = ${isArray ? 'texture(depthTex, vec3(clamp(uv, 0.0, 1.0), float(layer)))'
			: 'texture(depthTex, clamp(uv, 0.0, 1.0))'};
		float d = ${depthDecode(format)};
		metres = zbParams.x / (d * 2.0 - 1.0 + zbParams.y);
	} else {
		vec2 e = texture(envTex, (envRect.xy + viewUv * envRect.zw) / envSize).rg;
		// Coverage is 0..1; stretch it across the whole ramp to read it.
		metres = (stage == 1) ? e.r : e.g * range;
	}

	vec3 col = turbo(clamp(metres / range, 0.0, 1.0));
	// A contour every 25 cm, so absolute distance is readable, not just relative.
	col *= mix(0.55, 1.0, smoothstep(0.0, 0.06, abs(fract(metres * 4.0) - 0.5)));
	fragColor = vec4(col, 1.0);
}
`;
}

const depthUniforms = {
	depthTex: { value: null },
	dstRect: { value: new THREE.Vector4() },
	zbParams: { value: new THREE.Vector2(-2 * 0.08, -1) },
	depthUvXform: { value: new THREE.Matrix4() },   // normDepthBufferFromNormView
	near: { value: 0.08 },
	far: { value: 0 },        // 0 means "infinite", Meta's documented case
	range: { value: P.depthrange },
	rawScale: { value: 1 },
	layer: { value: 0 },
	stage: { value: 0 },
	envTex: { value: null },
	envRect: { value: new THREE.Vector4() },
	envSize: { value: new THREE.Vector2(2, 2) },
};

// Meta's _EnvironmentDepthZBufferParams, derived from the projection. A far of
// 0 or non-finite means an infinite far plane, which is what the on-device
// measurements say this sensor actually uses.
function setZBufferParams(near, far) {
	const v = depthUniforms.zbParams.value;
	if (!Number.isFinite(far) || far <= 0) v.set(-2 * near, -1);
	else v.set(-2 * far * near / (far - near), -(far + near) / (far - near));
	depthUniforms.near.value = near;
	depthUniforms.far.value = far;
}

// ---------------------------------------------------------------- depth prepass

// The raw sensor map is coarse and, since we bind it ourselves, unfiltered.
// Resolving it once per frame into a small linear-metres buffer beats sampling
// it repeatedly: the filtering happens at a quarter resolution, the result gets
// hardware bilinear for free, and the volume shader is left with ONE fetch.
//
// The second channel is the local depth RANGE over the kernel. It is near zero
// on a flat surface and large across a silhouette, which is what lets the cloud
// bleed only at edges instead of glowing through the middle of a hand.
function envPrepassFragment(isArray, taps, filter) {
	return /* glsl */`
precision highp float;
${isArray ? 'precision highp sampler2DArray;\nuniform sampler2DArray depthTex;'
			: 'uniform sampler2D depthTex;'}
uniform vec4  dstRect;
uniform vec2  zbParams;
uniform mat4  depthUvXform;
uniform float envSoft;
uniform float occLo;
uniform float occHi;
uniform int   layer;

uniform mat4  invProj;      // clip -> view, for the per-pixel ray
uniform mat4  viewToTex;    // view -> volume local (directions only)
uniform vec3  camPos;       // eye in volume texture space
uniform vec3  bbMin;
uniform vec3  bbMax;
uniform vec3  mvZ;          // third row of modelView: metres of eye depth per t

out vec4 fragColor;

float envTexel(vec2 uv) {
	float d = ${isArray ? 'texture(depthTex, vec3(clamp(uv, 0.0, 1.0), float(layer))).r'
			: 'texture(depthTex, clamp(uv, 0.0, 1.0)).r'};
	return zbParams.x / (d * 2.0 - 1.0 + zbParams.y);
}

// The sensor texture belongs to the runtime and we bind it raw, so it carries no
// sampler state at all -- every fetch snaps to a texel centre and the filter has
// to be written out by hand.
${filter === 1 ? `
// Cubic B-spline. C2, so neither a derivative crease (plain bilinear) nor a flat
// spot at each texel centre (smoothstep weights) -- both of those put a visible
// signature on the sample lattice, as creases or as rounded blobs. This basis
// does nothing special at sample positions, so the grid leaves no trace.
//
// It is approximating rather than interpolating: it does not pass through the
// sampled values, it smooths them. Usually a drawback, here the point. And all
// four weights are non-negative, so it cannot overshoot -- Catmull-Rom is
// sharper but rings, and it would ring hardest at depth discontinuities, which
// is exactly where a halo of wrong distance would hurt most.
float envSmooth(vec2 uv) {
	vec2 sz = vec2(textureSize(depthTex, 0).xy);
	vec2 tx = 1.0 / sz;
	vec2 p = uv * sz - 0.5;
	vec2 b = floor(p);
	vec2 f = p - b;
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	vec2 om = 1.0 - f;

	float wx[4];
	float wy[4];
	wx[0] = om.x * om.x * om.x / 6.0;
	wx[1] = (3.0 * f3.x - 6.0 * f2.x + 4.0) / 6.0;
	wx[2] = (-3.0 * f3.x + 3.0 * f2.x + 3.0 * f.x + 1.0) / 6.0;
	wx[3] = f3.x / 6.0;
	wy[0] = om.y * om.y * om.y / 6.0;
	wy[1] = (3.0 * f3.y - 6.0 * f2.y + 4.0) / 6.0;
	wy[2] = (-3.0 * f3.y + 3.0 * f2.y + 3.0 * f.y + 1.0) / 6.0;
	wy[3] = f3.y / 6.0;

	// Weights sum to one on each axis, so no normalisation afterwards.
	float acc = 0.0;
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			vec2 o = (b + vec2(float(i) - 1.0, float(j) - 1.0) + 0.5) * tx;
			acc += wx[i] * wy[j] * envTexel(o);
		}
	}
	return acc;
}` : `
// Smoothstep-weighted bilinear. Kept for comparison: it removes the bilinear
// crease but replaces it with a flat spot at every texel centre, which reads as
// rounded blobs under magnification.
float envSmooth(vec2 uv) {
	vec2 sz = vec2(textureSize(depthTex, 0).xy);
	vec2 p = uv * sz - 0.5;
	vec2 f = smoothstep(0.0, 1.0, fract(p));
	vec2 b = (floor(p) + 0.5) / sz;
	vec2 tx = 1.0 / sz;
	return mix(mix(envTexel(b), envTexel(b + vec2(tx.x, 0.0)), f.x),
	           mix(envTexel(b + vec2(0.0, tx.y)), envTexel(b + tx), f.x), f.y);
}`}

vec3 safeRcp(vec3 v) {
	vec3 s = vec3(v.x < 0.0 ? -1.0 : 1.0, v.y < 0.0 ? -1.0 : 1.0, v.z < 0.0 ? -1.0 : 1.0);
	return s / max(abs(v), vec3(1e-8));
}

vec2 boxIntersect(vec3 ro, vec3 rd, vec3 lo, vec3 hi) {
	vec3 inv = safeRcp(rd);
	vec3 t0 = (lo - ro) * inv;
	vec3 t1 = (hi - ro) * inv;
	vec3 tmn = min(t0, t1), tmx = max(t0, t1);
	return vec2(max(max(tmn.x, tmn.y), tmn.z), min(min(tmx.x, tmx.y), tmx.z));
}

void main() {
	vec2 viewUv = clamp((gl_FragCoord.xy - dstRect.xy) / dstRect.zw, 0.0, 1.0);
	vec2 uv = (depthUvXform * vec4(viewUv, 0.0, 1.0)).xy;
	vec2 tx = envSoft / vec2(textureSize(depthTex, 0).xy);

	// This pixel's ray, rebuilt from the projection so the prepass can answer
	// the same question the volume asks: where does the cloud sit along it?
	vec2 ndc = viewUv * 2.0 - 1.0;
	vec4 nearPt = invProj * vec4(ndc, -1.0, 1.0);
	vec3 rd = normalize(mat3(viewToTex) * normalize(nearPt.xyz / nearPt.w));
	vec2 t = boxIntersect(camPos, rd, bbMin, bbMax);
	float k = max(-dot(rd, mvZ), 1e-6);
	float zNear = max(t.x, 0.0) * k;      // cloud's near face, metres
	float zFar = t.y * k;                 // cloud's far face
	float span = max(zFar - zNear, 1e-4);

	// The whole point of doing it here: the OCCLUSION is decided per tap and the
	// RESULTS are averaged. Blurring the distance first and thresholding after
	// cannot work -- a threshold collapses a smooth ramp straight back into a
	// hard edge at one texel's granularity, which is what kept the silhouette
	// blocky however hard the depth was filtered.
	// Ordered centre-first, so a smaller kernel is a prefix of the same pattern:
	// 1 = centre only, 5 = quincunx, 9 = full 3x3. Each tap costs four fetches
	// unless the sampler is doing the filtering for us.
	const vec2 OFFS[9] = vec2[9](
		vec2(0.0, 0.0),
		vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0),
		vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0));

	float sum = 0.0, cov = 0.0;
	for (int i = 0; i < ${taps}; i++) {
		float z = envSmooth(uv + OFFS[i] * tx);
		sum += z;
		// How much of the cloud's depth extent this sample hides: 0 when the
		// surface is behind the cloud, 1 when it is in front of all of it.
		cov += clamp((zFar - z) / span, 0.0, 1.0);
	}
	cov /= float(${taps});

	// Hands have large silhouettes - bring the occlusion down so that anything
	// below occLo reads as nothing, rescaled back up. Purely aesthetic, and it
	// doubles as the crispness control: narrowing the occLo..occHi window
	// steepens the cut without reintroducing any lattice structure, because it
	// acts on a field that is already smooth.
	cov = clamp((cov - occLo) / max(occHi - occLo, 1e-4), 0.0, 1.0);

	fragColor = vec4(sum / float(${taps}), cov, 0.0, 1.0);
}
`;
}

// ---------------------------------------------------------------- occlusion

// A textured quad that hides behind the real world. The environment depth is
// sampled at the fragment's own screen position and compared with the
// fragment's distance from the eye; if something real is nearer, the fragment
// is dropped and passthrough shows through instead.
//
// The screen-space lookup assumes the depth map is aligned with the current
// view, which is only approximately true -- the depth frame was captured at a
// different pose than the one being rendered, and Meta's documentation says it
// should be reprojected with the supplied pose and FOV. Under head motion the
// silhouette will lag. Left uncorrected on purpose for now.
const OCC_VERT = /* glsl */`
out vec2 vUv;
out float vViewZ;
void main() {
	vUv = uv;
	vec4 mv = modelViewMatrix * vec4(position, 1.0);
	vViewZ = -mv.z;                 // distance along the view axis, metres
	gl_Position = projectionMatrix * mv;
}
`;

function occlusionFragment(isArray) {
	return /* glsl */`
precision highp float;
${isArray ? 'precision highp sampler2DArray;\nuniform sampler2DArray depthTex;'
			: 'uniform sampler2D depthTex;'}
uniform sampler2D map;
uniform vec4  dstRect;
uniform vec2  zbParams;
uniform mat4  depthUvXform;   // normDepthBufferFromNormView
uniform float occBias;
uniform float occSoft;        // tap spread, in depth texels
uniform int   useDepth;
uniform int   layer;
in vec2 vUv;
in float vViewZ;
out vec4 fragColor;

float envAt(vec2 uv) {
	float d = ${isArray ? 'texture(depthTex, vec3(clamp(uv, 0.0, 1.0), float(layer))).r'
			: 'texture(depthTex, clamp(uv, 0.0, 1.0)).r'};
	return zbParams.x / (d * 2.0 - 1.0 + zbParams.y);
}

void main() {
	vec3 rgb = texture(map, vUv).rgb;
	if (useDepth == 0) { fragColor = vec4(rgb, 1.0); return; }

	vec2 viewUv = clamp((gl_FragCoord.xy - dstRect.xy) / dstRect.zw, 0.0, 1.0);
	// The runtime tells us how its depth buffer maps onto this view; sampling
	// with raw screen coordinates and hoping is what made the map lag.
	vec2 uv = (depthUvXform * vec4(viewUv, 0.0, 1.0)).xy;

	// Four taps, each tested independently, and the RESULTS averaged. Filtering
	// the depth itself would blend a near surface with a far one and produce a
	// distance that exists nowhere -- the silhouette lands in the wrong place
	// and crawls. Averaging the binary test antialiases the edge honestly.
	vec2 texel = occSoft / vec2(textureSize(depthTex, 0).xy);
	float occ = 0.0;
	occ += (envAt(uv + vec2(-texel.x, -texel.y)) < vViewZ - occBias) ? 1.0 : 0.0;
	occ += (envAt(uv + vec2( texel.x, -texel.y)) < vViewZ - occBias) ? 1.0 : 0.0;
	occ += (envAt(uv + vec2(-texel.x,  texel.y)) < vViewZ - occBias) ? 1.0 : 0.0;
	occ += (envAt(uv + vec2( texel.x,  texel.y)) < vViewZ - occBias) ? 1.0 : 0.0;
	occ *= 0.25;

	if (occ > 0.999) discard;
	fragColor = vec4(rgb, 1.0 - occ);
}
`;
}

// ---------------------------------------------------------------- renderer

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearAlpha(0);
renderer.xr.enabled = true;
renderer.xr.setFramebufferScaleFactor(P.fbscale);
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.01, 50);

// Two scenes: the volume renders alone into a low-resolution buffer, everything
// else stays at full resolution. They share a transform.
const scene = new THREE.Scene();
scene.background = null;
const volumeScene = new THREE.Scene();

const cloud = new THREE.Group();          // axes and labels
const volumeGroup = new THREE.Group();    // the cube
for (const g of [cloud, volumeGroup]) {
	g.rotation.x = -Math.PI / 2;   // data Z (gravity) -> world Y (up)
	g.scale.setScalar(P.size);
}
scene.add(cloud);
volumeScene.add(volumeGroup);

// The cube lives in two scenes and its caption in a third place, so where it is
// has to be shared state rather than three copies that can drift apart.
const cloudPos = new THREE.Vector3(0, P.height, -P.dist);

function applyCloudPos() {
	cloud.position.copy(cloudPos);
	volumeGroup.position.copy(cloudPos);
	if (panel) panel.position.set(cloudPos.x, cloudPos.y + P.size * 0.95, cloudPos.z);
	// TEMPORARY: the banner rides along with the volume so it can be dragged
	// into the open instead of being read through a wall. Revert to a fixed
	// world position once the depth work is done.
	if (helpMesh) helpMesh.position.copy(cloudPos).add(helpOffset);
	positionLabels();
}

const uniforms = {
	volume: { value: null },
	bbMin: { value: new THREE.Vector3(0, 0, 0) },
	bbMax: { value: new THREE.Vector3(1, 1, 1) },
	camPos: { value: new THREE.Vector3() },
	densityGain: { value: P.gain },
	threshold: { value: P.threshold },
	stepLen: { value: SQRT3 / P.steps },
	maxSteps: { value: P.steps },
	lightMode: { value: P.light ? 1 : 0 },
	// Environment occlusion. depthTex/zbParams/depthUvXform are shared by
	// reference with the depth overlay so all three always agree.
	envTex: { value: null },
	envRect: { value: new THREE.Vector4() },
	envSize: { value: new THREE.Vector2(2, 2) },
	dstRect: { value: new THREE.Vector4() },
	useDepth: { value: 0 },
};

let steps = P.steps;
let thresholdBoxes = null;
let bbDiag = SQRT3;

function setSteps(n) {
	steps = Math.min(1024, Math.max(4, Math.round(n)));
	uniforms.maxSteps.value = steps;
	// `steps` is the count a ray crossing the whole marched box takes; shorter
	// rays finish early. Tying it to the tight box rather than the cube is what
	// turns a higher threshold into finer sampling instead of wasted steps.
	uniforms.stepLen.value = bbDiag / steps;
}

// Point the marcher at the box for the current threshold.
function applyThreshold() {
	const t = uniforms.threshold.value;
	// The shader keeps a voxel when d > threshold, i.e. v > 255*t, i.e. v >= j.
	const j = Math.floor(t * 255) + 1;
	const box = (thresholdBoxes && j <= 255) ? thresholdBoxes[j] : undefined;

	if (box) {
		uniforms.bbMin.value.fromArray(box.min);
		uniforms.bbMax.value.fromArray(box.max);
		bbDiag = box.diag;
	} else if (thresholdBoxes) {
		uniforms.bbMin.value.set(0, 0, 0);   // nothing survives; every ray misses
		uniforms.bbMax.value.set(0, 0, 0);
		bbDiag = SQRT3;
	} else {
		uniforms.bbMin.value.set(0, 0, 0);   // boxes not built yet
		uniforms.bbMax.value.set(1, 1, 1);
		bbDiag = SQRT3;
	}
	setSteps(steps);
}
setSteps(P.steps);

function makeVolumeMaterial() {
	return new THREE.ShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms,
		vertexShader: VOL_VERT,
		fragmentShader: volumeFragment(),
		side: THREE.BackSide,      // fragments survive when the head is inside the box
		transparent: true,
		depthTest: false,
		depthWrite: false,
		premultipliedAlpha: true,
	});
}

const volumeMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), makeVolumeMaterial());
volumeMesh.frustumCulled = false;

// three calls this once per view, so each eye gets its own camera position --
// and its own modelView, which is what mvZ carries.
const _eye = new THREE.Vector3();
volumeMesh.onBeforeRender = (r, s, cam) => {
	_eye.setFromMatrixPosition(cam.matrixWorld);
	volumeMesh.worldToLocal(_eye);
	uniforms.camPos.value.copy(_eye).addScalar(0.5);

	// The volume is drawn into the low-resolution buffer with the viewport
	// scaled by rtScale, so gl_FragCoord spans the SCALED rectangle -- unlike
	// the banner, which is drawn into the XR framebuffer at full size. Using
	// cam.viewport unscaled here made the shader read the depth map from
	// entirely the wrong part of the screen whenever rtScale was below 1.
	const vs = renderer.getRenderTarget() === rt ? rtScale : 1;
	if (cam.viewport) {
		const v = cam.viewport;
		uniforms.dstRect.value.set(
			Math.floor(v.x * vs), Math.floor(v.y * vs),
			Math.floor(v.z * vs), Math.floor(v.w * vs));
		const eyes = renderer.xr.getCamera()?.cameras;
		const i = eyes ? eyes.indexOf(cam) : 0;
		uniforms.envRect.value.copy(envRects[i < 0 ? 0 : i]);
	} else {
		renderer.getDrawingBufferSize(_size);
		uniforms.dstRect.value.set(0, 0, Math.floor(_size.x * vs), Math.floor(_size.y * vs));
		uniforms.envRect.value.copy(envRects[0]);
	}
	uniforms.envSize.value.set(envRT.width, envRT.height);
};
volumeGroup.add(volumeMesh);

// ---------------------------------------------------------------- volume buffer

const rt = new THREE.WebGLRenderTarget(2, 2, {
	depthBuffer: false,
	stencilBuffer: false,
	format: THREE.RGBAFormat,
	type: THREE.UnsignedByteType,
});
rt.texture.minFilter = rt.texture.magFilter = THREE.LinearFilter;
rt.texture.generateMipmaps = false;
rt.texture.colorSpace = THREE.NoColorSpace;

let rtScale = P.rtscale;

const blitUniforms = {
	tex: { value: rt.texture },
	dstRect: { value: new THREE.Vector4() },
	srcRect: { value: new THREE.Vector4() },
	texSize: { value: new THREE.Vector2(2, 2) },
};

const blitQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
	glslVersion: THREE.GLSL3,
	uniforms: blitUniforms,
	vertexShader: BLIT_VERT,
	fragmentShader: BLIT_FRAG,
	transparent: true,
	premultipliedAlpha: true,
	depthTest: false,
	depthWrite: false,
}));
blitQuad.frustumCulled = false;
blitQuad.renderOrder = 100;   // over the axes, as render3.py drew it

const _vp = new THREE.Vector4();
const _size = new THREE.Vector2();

blitQuad.onBeforeRender = (r, s, cam) => {
	if (cam.viewport) {
		_vp.copy(cam.viewport);
	} else {
		renderer.getDrawingBufferSize(_size);
		_vp.set(0, 0, _size.x, _size.y);
	}
	blitUniforms.dstRect.value.copy(_vp);
	blitUniforms.srcRect.value.set(
		Math.floor(_vp.x * rtScale), Math.floor(_vp.y * rtScale),
		Math.floor(_vp.z * rtScale), Math.floor(_vp.w * rtScale));
	blitUniforms.texSize.value.set(rt.width, rt.height);
};

// Sized to the FULL framebuffer and left alone. rtScale only chooses how much
// of it we render into and read back from.
//
// rt.setSize() disposes the GL texture and allocates a new one; doing that
// every time the controller nudged the scale meant freeing a multi-megabyte
// target while the previous frame's blit might still be reading it -- exactly
// the situation where tile-based mobile GPUs hand back a frame of stale
// content. Reallocating only on a genuine framebuffer size change also makes
// scale adjustment free, so the controller can move in small steps often
// instead of big steps rarely.
function resizeRT(w, h) {
	w = Math.max(2, Math.floor(w));
	h = Math.max(2, Math.floor(h));
	if (rt.width !== w || rt.height !== h) {
		rt.setSize(w, h);
		console.log(`[xrviz] volume buffer ${w}x${h} (full framebuffer; rtScale scales the viewport)`);
	}
}

// Renders the volume alone into `rt`, one pass per eye in XR.
function renderVolumePass() {
	const xr = renderer.xr;
	const autoClear = renderer.autoClear;

	// Must be restored, never assumed to be null. In an XR frame WebXRManager
	// has already bound its own render target wrapping XRWebGLLayer.framebuffer,
	// and setRenderTarget(null) would hand the canvas back instead -- the whole
	// stereo frame then lands on the page and the headset stays black. The
	// emulator hides this because its layer framebuffer *is* the default one.
	const prevRT = renderer.getRenderTarget();

	if (xr.isPresenting) {
		const xrCam = xr.getCamera();
		let fbW = 0, fbH = 0;
		for (const c of xrCam.cameras) {
			fbW = Math.max(fbW, c.viewport.x + c.viewport.z);
			fbH = Math.max(fbH, c.viewport.y + c.viewport.w);
		}
		resizeRT(fbW, fbH);

		// Otherwise three binds the XR framebuffer and its own camera.
		xr.enabled = false;
		renderer.autoClear = false;
		renderer.setRenderTarget(rt);
		renderer.setScissorTest(false);
		renderer.clear(true, false, false);
		for (const c of xrCam.cameras) {
			const v = c.viewport;
			const x = Math.floor(v.x * rtScale), y = Math.floor(v.y * rtScale);
			const w = Math.floor(v.z * rtScale), h = Math.floor(v.w * rtScale);
			renderer.setViewport(x, y, w, h);
			renderer.setScissor(x, y, w, h);
			renderer.setScissorTest(true);
			renderer.render(volumeScene, c);
		}
		renderer.setScissorTest(false);
		// Never restore a target the session may have taken with it.
		renderer.setRenderTarget(xr.isPresenting ? prevRT : null);
		renderer.setViewport(0, 0, fbW, fbH);
		xr.enabled = true;
		renderer.autoClear = autoClear;
	} else {
		renderer.getDrawingBufferSize(_size);
		resizeRT(_size.x, _size.y);
		renderer.autoClear = false;
		renderer.setRenderTarget(rt);
		renderer.setScissorTest(false);
		renderer.clear(true, false, false);
		renderer.setViewport(0, 0, Math.floor(_size.x * rtScale), Math.floor(_size.y * rtScale));
		renderer.render(volumeScene, camera);
		// Never restore a target the session may have taken with it.
		renderer.setRenderTarget(xr.isPresenting ? prevRT : null);
		renderer.setViewport(0, 0, _size.x, _size.y);
		renderer.autoClear = autoClear;
	}
}

// ---------------------------------------------------------------- depth prepass buffer

// Quarter resolution: the sensor is coarser than that anyway, the filtering
// costs a sixteenth of what it would at full size, and the upsample back to
// screen resolution is itself another round of smoothing, for free.
const envScale = P.envscale;
const envHalfFloat = !!(renderer.getContext().getExtension('EXT_color_buffer_half_float')
	|| renderer.getContext().getExtension('EXT_color_buffer_float'));

const envRT = new THREE.WebGLRenderTarget(2, 2, {
	depthBuffer: false, stencilBuffer: false,
	format: THREE.RGBAFormat,
	type: envHalfFloat ? THREE.HalfFloatType : THREE.UnsignedByteType,
});
envRT.texture.minFilter = envRT.texture.magFilter = THREE.LinearFilter;
envRT.texture.wrapS = envRT.texture.wrapT = THREE.ClampToEdgeWrapping;
envRT.texture.generateMipmaps = false;
envRT.texture.colorSpace = THREE.NoColorSpace;
uniforms.envTex.value = envRT.texture;

const envUniforms = {
	depthTex: depthUniforms.depthTex,
	zbParams: depthUniforms.zbParams,
	depthUvXform: depthUniforms.depthUvXform,
	dstRect: { value: new THREE.Vector4() },
	envSoft: { value: P.envsoft },
	occLo: { value: P.occlo },
	occHi: { value: P.occhi },
	layer: { value: 0 },
	// The prepass now needs the same geometry the volume uses, so it can work
	// out where the cloud lies along each pixel's ray.
	invProj: { value: new THREE.Matrix4() },
	viewToTex: { value: new THREE.Matrix4() },
	camPos: { value: new THREE.Vector3() },
	bbMin: uniforms.bbMin,
	bbMax: uniforms.bbMax,
	mvZ: { value: new THREE.Vector3() },
};

const _envMv = new THREE.Matrix4();
const _envEye = new THREE.Vector3();

// Per eye: the ray reconstruction matrices and the cloud's placement.
function envCameraUniforms(cam) {
	volumeMesh.updateWorldMatrix(true, false);   // the volume scene renders later
	_envMv.multiplyMatrices(cam.matrixWorldInverse, volumeMesh.matrixWorld);
	const e = _envMv.elements;
	envUniforms.mvZ.value.set(e[2], e[6], e[10]);
	envUniforms.viewToTex.value.copy(_envMv).invert();
	envUniforms.invProj.value.copy(cam.projectionMatrixInverse);
	_envEye.setFromMatrixPosition(cam.matrixWorld);
	volumeMesh.worldToLocal(_envEye);
	envUniforms.camPos.value.copy(_envEye).addScalar(0.5);
}

const envScene = new THREE.Scene();
const envCam = new THREE.PerspectiveCamera();
const envQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial());
envQuad.frustumCulled = false;
envScene.add(envQuad);

// Where each eye landed inside envRT, so the volume can find its own half.
const envRects = [new THREE.Vector4(), new THREE.Vector4()];

function resizeEnv(w, h) {
	w = Math.max(2, Math.floor(w));
	h = Math.max(2, Math.floor(h));
	if (envRT.width !== w || envRT.height !== h) {
		envRT.setSize(w, h);
		console.log(`[xrviz] env buffer ${w}x${h} (envscale=${envScale})`);
	}
}

// Resolves the sensor map into linear metres plus an edge signal, once per eye.
function renderEnvPass() {
	const xr = renderer.xr;
	const prevRT = renderer.getRenderTarget();
	const autoClear = renderer.autoClear;
	const wasEnabled = xr.enabled;

	const draw = (cam, rect, layer, i) => {
		renderer.setViewport(rect.x, rect.y, rect.z, rect.w);
		renderer.setScissor(rect.x, rect.y, rect.z, rect.w);
		renderer.setScissorTest(true);      // keeps one eye out of the other
		envUniforms.dstRect.value.copy(rect);
		envUniforms.layer.value = layer;
		envCameraUniforms(cam);
		renderer.render(envScene, envCam);
		envRects[i].copy(rect);
	};

	try {
		xr.enabled = false;
		renderer.autoClear = false;
		if (wasEnabled && xr.isPresenting) {
			const xrCam = xr.getCamera();
			let fbW = 0, fbH = 0;
			for (const c of xrCam.cameras) {
				fbW = Math.max(fbW, c.viewport.x + c.viewport.z);
				fbH = Math.max(fbH, c.viewport.y + c.viewport.w);
			}
			resizeEnv(fbW * envScale, fbH * envScale);
			renderer.setRenderTarget(envRT);
			renderer.setScissorTest(false);
			renderer.clear(true, false, false);
			xrCam.cameras.forEach((c, i) => {
				const v = c.viewport;
				draw(c, new THREE.Vector4(
					Math.floor(v.x * envScale), Math.floor(v.y * envScale),
					Math.floor(v.z * envScale), Math.floor(v.w * envScale)), i, i);
			});
		} else {
			renderer.getDrawingBufferSize(_size);
			resizeEnv(_size.x * envScale, _size.y * envScale);
			renderer.setRenderTarget(envRT);
			renderer.setScissorTest(false);
			renderer.clear(true, false, false);
			draw(camera, new THREE.Vector4(0, 0, envRT.width, envRT.height), 0, 0);
		}
	} finally {
		renderer.setScissorTest(false);
		// Never restore a target the session may have taken with it.
		renderer.setRenderTarget(xr.isPresenting ? prevRT : null);
		renderer.autoClear = autoClear;
		xr.enabled = wasEnabled;
	}
}

// ---------------------------------------------------------------- axes

const AXES = {
	X: { dir: [1, 0, 0], dark: [0.95, 0.35, 0.35], light: [0.78, 0.12, 0.12] },
	Y: { dir: [0, 1, 0], dark: [0.35, 0.90, 0.45], light: [0.05, 0.52, 0.20] },
	Z: { dir: [0, 0, 1], dark: [0.45, 0.60, 1.00], light: [0.15, 0.28, 0.80] },
};

function buildAxes(light) {
	const pos = [], col = [];
	for (const { dir, dark, light: lc } of Object.values(AXES)) {
		const c = light ? lc : dark;
		const dim = light ? c.map(v => v * 0.3 + 0.7) : c.map(v => v * 0.35);
		const seg = (a, b, k) => {
			pos.push(...dir.map(d => d * a), ...dir.map(d => d * b));
			col.push(...k, ...k);
		};
		seg(-0.55, 0.0, dim);
		seg(0.0, 0.70, c);
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
	const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
		vertexColors: true, transparent: true, depthTest: false, depthWrite: false,
	}));
	lines.renderOrder = 1;
	lines.frustumCulled = false;
	return lines;
}

// A plane rather than a Sprite, and deliberately not billboarded at all.
// Sprites turn to follow your head, and anything that moves by itself is
// unpleasant in VR. The orientation is set once, when the cloud is placed, and
// then stays put with the axes -- which are self-explanatory anyway.
// A quad that disappears when you walk behind it is a hole in the world.
// DoubleSide would keep it visible but show the text mirrored, which is no
// better. Two back-to-back single-sided copies read correctly from either side,
// and exactly one of them survives backface culling in any given view.
function twoSided(mesh) {
	const back = mesh.clone();          // shares geometry and material
	back.rotation.y += Math.PI;
	const g = new THREE.Group();
	g.add(mesh, back);
	return g;
}

function labelPlane(text, rgb) {
	const s = 128;
	const cv = document.createElement('canvas');
	cv.width = cv.height = s;
	const g = cv.getContext('2d');
	g.fillStyle = `rgb(${rgb.map(v => Math.round(v * 255)).join(',')})`;
	g.font = 'bold 84px DejaVu Sans, sans-serif';
	g.textAlign = 'center';
	g.textBaseline = 'middle';
	g.fillText(text, s / 2, s / 2);
	const tex = new THREE.CanvasTexture(cv);
	tex.colorSpace = THREE.SRGBColorSpace;
	tex.minFilter = THREE.LinearFilter;
	tex.generateMipmaps = false;
	const w = 0.14 * P.size;
	const m = new THREE.Mesh(new THREE.PlaneGeometry(w, w), new THREE.MeshBasicMaterial({
		map: tex, transparent: true, depthTest: false, depthWrite: false,
	}));
	m.renderOrder = 110;   // after the volume composite, as render3.py drew it
	return twoSided(m);
}

const axesDark = buildAxes(false), axesLight = buildAxes(true);
cloud.add(axesDark, axesLight);

// Labels live in world space, not inside `cloud`: that group is rotated -90
// degrees about X to stand the data up, and a flat quad parented to it would
// inherit the tilt. This is the same mapping applied by hand.
const AXIS_WORLD = { X: [1, 0, 0], Y: [0, 0, -1], Z: [0, 1, 0] };

// Each label lies in the plane containing its own axis, so it reads as part of
// the axis rather than as something aimed at the viewer. Fixed once, identical
// in VR and on the desktop, and it never turns.
const AXIS_YAW = { X: 0, Y: Math.PI / 2, Z: 0 };

const labelGroup = new THREE.Group();
scene.add(labelGroup);

const labels = { dark: [], light: [] };
for (const [name, a] of Object.entries(AXES)) {
	for (const mode of ['dark', 'light']) {
		const m = labelPlane(name, a[mode]);
		m.userData.dir = AXIS_WORLD[name];
		m.rotation.set(0, AXIS_YAW[name], 0);
		labels[mode].push(m);
		labelGroup.add(m);
	}
}

function positionLabels() {
	const r = P.size * 0.78;
	for (const m of labelGroup.children) {
		const d = m.userData.dir;
		m.position.set(cloudPos.x + d[0] * r, cloudPos.y + d[1] * r, cloudPos.z + d[2] * r);
	}
}


// ---------------------------------------------------------------- backdrop

// Quest composites immersive-ar with alpha-blend, so anything opaque we draw
// hides the passthrough behind it. A big inward-facing sphere is therefore a
// "VR mode" switch that needs no second session -- and switching session mode
// for real would need user activation, which a polled thumbstick cannot give.
const backdrop = new THREE.Mesh(
	new THREE.SphereGeometry(12, 24, 16),
	new THREE.MeshBasicMaterial({
		color: 0x05050a, side: THREE.BackSide, depthTest: false, depthWrite: false,
	}));
backdrop.renderOrder = -10;
backdrop.frustumCulled = false;
backdrop.visible = false;
scene.add(backdrop);

function toggleBackdrop() {
	backdrop.visible = !backdrop.visible;
	updateHud();
}

// ---------------------------------------------------------------- readout panel

// Sits above the cube like a caption on an exhibit. Yaw-only billboarding, so
// it is readable from any side but never pitches or rolls with your head --
// things that swim with your gaze are what makes VR unpleasant.
const panelCv = document.createElement('canvas');
panelCv.width = 512;
panelCv.height = 200;   // five lines: the depth readout adds two
const panelTex = new THREE.CanvasTexture(panelCv);
panelTex.colorSpace = THREE.SRGBColorSpace;
panelTex.minFilter = THREE.LinearFilter;
panelTex.generateMipmaps = false;

const panel = new THREE.Mesh(
	new THREE.PlaneGeometry(P.size * 1.1, P.size * 1.1 * panelCv.height / panelCv.width),
	new THREE.MeshBasicMaterial({
		map: panelTex, transparent: true, depthTest: false, depthWrite: false,
	}));
panel.renderOrder = 120;
panel.position.set(0, P.height + P.size * 0.95, -P.dist);
scene.add(panel);

function drawPanel(lines) {
	const g = panelCv.getContext('2d');
	const light = uniforms.lightMode.value === 1;
	g.clearRect(0, 0, panelCv.width, panelCv.height);
	g.fillStyle = light ? 'rgba(245,245,255,0.75)' : 'rgba(10,10,18,0.60)';
	g.fillRect(0, 0, panelCv.width, panelCv.height);
	g.font = '14px DejaVu Sans Mono, monospace';
	g.textBaseline = 'top';
	g.fillStyle = light ? '#20202a' : '#dcdce8';
	lines.forEach((ln, i) => g.fillText(ln, 14, 12 + i * 34));
	panelTex.needsUpdate = true;
}

const _pw = new THREE.Vector3(), _cw = new THREE.Vector3();

function facePanel(cam) {
	panel.getWorldPosition(_pw);
	_cw.setFromMatrixPosition(cam.matrixWorld);
	const dx = _cw.x - _pw.x, dz = _cw.z - _pw.z;
	if (dx * dx + dz * dz > 1e-8) panel.rotation.set(0, Math.atan2(dx, dz), 0);
}

// ---------------------------------------------------------------- instructions

// One source of truth per language: VR_inputs.<LANG>.txt is served as a static
// file, drawn onto a quad for XR and dumped into the DOM for the flat page.
let helpMesh = null;
let helpFace = null;      // the front quad, whose material carries the occlusion
let helpMap = null;
const helpOffset = new THREE.Vector3();

function makeOcclusionMaterial(isArray) {
	return new THREE.ShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms: {
			// Shared by reference with the depth overlay, so both always agree.
			depthTex: depthUniforms.depthTex,
			zbParams: depthUniforms.zbParams,
			depthUvXform: depthUniforms.depthUvXform,
			map: { value: helpMap },
			dstRect: { value: new THREE.Vector4() },
			occBias: { value: P.occbias },
			occSoft: { value: P.occsoft },
			useDepth: { value: 0 },
			layer: { value: 0 },
		},
		vertexShader: OCC_VERT,
		fragmentShader: occlusionFragment(isArray),
		// Transparent so the averaged coverage can feather the silhouette
		// instead of every fragment being all-or-nothing.
		transparent: true,
		depthTest: false,
		depthWrite: false,
	});
}

async function buildHelp() {
	let txt;
	try {
		const r = await fetch(T.helpFile);
		if (!r.ok) throw new Error(`${r.status}`);
		txt = await r.text();
	} catch (e) {
		console.warn(`${T.helpFile} not readable:`, e.message);
		return;
	}
	document.getElementById('help').textContent = txt;

	const lines = txt.replace(/\t/g, '    ').replace(/\s+$/, '').split('\n');
	const FS = 26, LH = 34, PAD = 30;
	const cv = document.createElement('canvas');
	cv.width = 960;
	cv.height = PAD * 2 + lines.length * LH;
	const g = cv.getContext('2d');
	g.fillStyle = '#000000';        // fully opaque, so it reads as a solid panel
	g.fillRect(0, 0, cv.width, cv.height);
	g.font = `${FS}px DejaVu Sans Mono, monospace`;
	g.textBaseline = 'top';
	lines.forEach((ln, i) => {
		// \p{Lu} rather than A-Z: "WYJŚCIE:" is a heading too, and an ASCII class
		// silently drops every heading that happens to carry a diacritic.
		g.fillStyle = /^\s*(==|[\p{Lu} ]+:)/u.test(ln) ? '#ffd479' : '#d6d6e4';
		g.fillText(ln, PAD, PAD + i * LH);
	});

	const tex = new THREE.CanvasTexture(cv);
	tex.colorSpace = THREE.SRGBColorSpace;
	tex.minFilter = THREE.LinearFilter;
	tex.generateMipmaps = false;
	helpMap = tex;

	const w = 1.7, h = w * cv.height / cv.width;
	const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), makeOcclusionMaterial(false));
	m.renderOrder = 90;                                 // the cloud draws over it
	m.onBeforeRender = (r, s, cam) => {
		const u = m.material.uniforms;
		if (cam.viewport) {
			u.dstRect.value.copy(cam.viewport);
			const eyes = renderer.xr.getCamera()?.cameras;
			const i = eyes ? eyes.indexOf(cam) : 0;
			u.layer.value = i < 0 ? 0 : i;
		} else {
			renderer.getDrawingBufferSize(_size);
			u.dstRect.value.set(0, 0, _size.x, _size.y);
			u.layer.value = 0;
		}
	};
	helpFace = m;
	helpMesh = twoSided(m);                             // readable from either side
	helpMesh.position.set(0, P.height, P.dist * 1.3);   // behind the start position
	helpMesh.rotation.y = Math.PI;                      // facing back at the origin
	helpOffset.copy(helpMesh.position).sub(cloudPos);
	scene.add(helpMesh);
}

// ---------------------------------------------------------------- depth overlay

// Everything here is opt-in and fails soft: if the runtime does not grant
// depth-sensing, the overlay simply never shows and nothing else in the scene
// notices. ?depth=0 stops us even asking for it.
const depthQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial());
depthQuad.frustumCulled = false;
depthQuad.renderOrder = 140;   // over everything, it is a diagnostic view
depthQuad.visible = false;
scene.add(depthQuad);

let depthView = false;      // effective: held on the controller, or latched by key
let depthKey = false;       // keyboard latch
let depthStage = 0;         // which raster of the pipeline to look at
let depthNote = '';
let depthBinding = null;    // XRWebGLBinding, one per session
let depthExternal = null;   // stable THREE.ExternalTexture we re-point each frame
let depthMeta = null;       // what the runtime actually gave us
let depthIsArray = null;

// three r180 binds an ExternalTexture correctly for sampler2D but not for
// sampler2DArray: setTexture2DArray has no isExternalTexture branch, and an
// ExternalTexture has version 0 so the upload path is skipped too -- it ends up
// binding undefined, and every sample reads 0. Pointing three's texture
// properties at the real GL texture ourselves is the whole workaround.
function bindExternal(tex, glTexture) {
	renderer.properties.get(tex).__webglTexture = glTexture;
}

let depthFormat = null;

function ensureDepthMaterial(isArray, format) {
	if (depthIsArray === isArray && depthFormat === format) return;
	depthIsArray = isArray;
	depthFormat = format;
	depthQuad.material?.dispose?.();
	depthQuad.material = new THREE.ShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms: depthUniforms,
		vertexShader: BLIT_VERT,
		fragmentShader: depthFragment(isArray, format),
		depthTest: false,
		depthWrite: false,
	});
	// The prepass is what reads the sensor, so its sampler type must match.
	envQuad.material?.dispose?.();
	envQuad.material = new THREE.ShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms: envUniforms,
		vertexShader: BLIT_VERT,
		fragmentShader: envPrepassFragment(isArray,
			[1, 5, 9].includes(P.envtaps) ? P.envtaps : 1, P.envfilter),
		depthTest: false,
		depthWrite: false,
	});

	// The banner samples the same texture, so its sampler type must match too.
	if (helpFace) {
		const old = helpFace.material;
		helpFace.material = makeOcclusionMaterial(isArray);
		helpFace.material.uniforms.map.value = helpMap;
		helpFace.material.uniforms.occBias.value = old.uniforms.occBias.value;
		helpFace.material.uniforms.useDepth.value = 1;
		helpMesh?.children.forEach(c => (c.material = helpFace.material));
		old.dispose();
	}
	console.log(`[xrviz] depth shader: ${isArray ? 'texture-array' : 'texture'} / ${format}`);
}

// ---------------------------------------------------------------- depth probe

// Reads the four raw channels at the centre of the view back to the CPU and
// puts the numbers on the panel. Guessing the encoding from how the picture
// looks has now cost two trips to the headset; four integers and a known
// distance settle it arithmetically.
// Reads the middle of the view back to the CPU: the prepass has already turned
// the sensor into linear metres, so this is a distance in metres at the point
// you are looking at -- a depth ruler, once it is put on screen.
const PROBE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D envTex;
uniform vec4 envRect;
uniform vec2 envSize;
out vec4 fragColor;
void main() {
	fragColor = texture(envTex, (envRect.xy + 0.5 * envRect.zw) / envSize);
}
`;

const probeFloat = !!renderer.getContext().getExtension('EXT_color_buffer_float');
const probeRT = new THREE.WebGLRenderTarget(1, 1, {
	depthBuffer: false, stencilBuffer: false,
	format: THREE.RGBAFormat,
	type: probeFloat ? THREE.FloatType : THREE.UnsignedByteType,
});
const probeScene = new THREE.Scene();
const probeCam = new THREE.PerspectiveCamera();
const probeMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
	glslVersion: THREE.GLSL3,
	uniforms: depthUniforms,
	vertexShader: BLIT_VERT,
	fragmentShader: PROBE_FRAG,
	depthTest: false,
	depthWrite: false,
}));
probeMesh.frustumCulled = false;
probeScene.add(probeMesh);

const probeBuf = probeFloat ? new Float32Array(4) : new Uint8Array(4);
let probeMetres = -1;
let probeDue = 0;

function probeDepth(now) {
	if (now < probeDue || !depthUniforms.envTex.value) return;
	probeDue = now + 300;   // readPixels stalls the pipeline; do it rarely
	const xr = renderer.xr;
	const wasEnabled = xr.enabled;
	const prevRT = renderer.getRenderTarget();
	const autoClear = renderer.autoClear;
	try {
		xr.enabled = false;
		renderer.autoClear = true;
		renderer.setRenderTarget(probeRT);
		renderer.setViewport(0, 0, 1, 1);
		renderer.render(probeScene, probeCam);
		renderer.readRenderTargetPixels(probeRT, 0, 0, 1, 1, probeBuf);
		probeMetres = probeBuf[0];   // prepass red channel is already metres
	} catch (e) {
		probeMetres = -1;
	} finally {
		// Never restore a target the session may have taken with it.
		renderer.setRenderTarget(xr.isPresenting ? prevRT : null);
		renderer.autoClear = autoClear;
		xr.enabled = wasEnabled;
	}
}

depthQuad.onBeforeRender = (r, s, cam) => {
	if (cam.viewport) {
		depthUniforms.dstRect.value.copy(cam.viewport);
		const eyes = renderer.xr.getCamera()?.cameras;
		const i = eyes ? eyes.indexOf(cam) : 0;
		depthUniforms.layer.value = i < 0 ? 0 : i;   // one array layer per eye
		depthUniforms.envRect.value.copy(envRects[i < 0 ? 0 : i]);
	} else {
		renderer.getDrawingBufferSize(_size);
		depthUniforms.dstRect.value.set(0, 0, _size.x, _size.y);
		depthUniforms.layer.value = 0;
		depthUniforms.envRect.value.copy(envRects[0]);
	}
	depthUniforms.envTex.value = envRT.texture;
	depthUniforms.envSize.value.set(envRT.width, envRT.height);
};

// Talks to WebXR directly rather than through three's depth module: we need the
// metadata (textureType, rawValueToMeters, the real near/far) to interpret the
// values at all, and three exposes none of it.
function setUseDepth(on) {
	const u = helpFace?.material?.uniforms;
	if (u) u.useDepth.value = on ? 1 : 0;
	uniforms.useDepth.value = (on && P.volocc) ? 1 : 0;
}

// Runs every frame, not just when the overlay is on: the banner's occlusion
// needs the depth map whether or not anyone is looking at the depth map.
function updateDepth() {
	depthQuad.visible = false;

	const session = renderer.xr.getSession();
	if (!session) { depthNote = T.depthArOnly; setUseDepth(false); return; }
	if (!session.enabledFeatures?.includes('depth-sensing')) {
		depthNote = T.depthUnavailable;
		setUseDepth(false);
		return;
	}

	try {
		const frame = renderer.xr.getFrame();
		const refSpace = renderer.xr.getReferenceSpace();
		const pose = frame?.getViewerPose?.(refSpace);
		const view = pose?.views?.[0];
		if (!view) { depthNote = T.depthNoPose; setUseDepth(false); return; }

		if (!depthBinding) depthBinding = new XRWebGLBinding(session, renderer.getContext());
		const info = depthBinding.getDepthInformation(view);
		if (!info?.texture) { depthNote = T.depthNoFrame; setUseDepth(false); return; }

		const isArray = info.textureType === 'texture-array';
		ensureDepthMaterial(isArray, session.depthDataFormat ?? 'float32');
		if (!depthExternal) depthExternal = new THREE.ExternalTexture(info.texture);
		depthExternal.sourceTexture = info.texture;
		bindExternal(depthExternal, info.texture);

		depthUniforms.depthTex.value = depthExternal;
		depthUniforms.rawScale.value = info.rawValueToMeters ?? 1;

		// Re-read every frame: this is the runtime's own statement of how the
		// depth buffer lines up with the view it just gave us, and it is what
		// carries any pose correction. Identity is only correct by accident.
		const m = info.normDepthBufferFromNormView?.matrix;
		if (m) depthUniforms.depthUvXform.value.fromArray(m);

		// Deliberately NOT session.renderState.depthNear: three sets that from
		// our own camera (0.01), which is the rendering frustum, not the depth
		// sensor's projection. Only info.depthNear would be authoritative, and
		// only if it looks like a sensor near plane at all.
		const reported = info.depthNear;
		const sane = Number.isFinite(reported) && reported > 0.02 && reported < 1;
		setZBufferParams(P.depthnear > 0 ? P.depthnear : (sane ? reported : 0.08),
			Number.isFinite(info.depthFar) ? info.depthFar : 0);
		depthQuad.visible = depthView;
		depthNote = '';
		setUseDepth(true);   // depth is live this frame

		if (!depthMeta) {
			depthMeta = {
				textureType: info.textureType ?? 'texture',
				size: `${info.width}x${info.height}`,
				rawValueToMeters: info.rawValueToMeters,
				near: depthUniforms.near.value,
				far: depthUniforms.far.value,
				format: session.depthDataFormat,
				usage: session.depthUsage,
				hasUvXform: !!info.normDepthBufferFromNormView,
			};
			console.log('[xrviz] depth:', JSON.stringify(depthMeta));

			// The whole chain, so it is obvious which link is the coarse one.
			const eye = envRects[0];
			console.log('[xrviz] resolution chain: '
				+ `sensor ${info.width}x${info.height} per eye`
				+ ` -> env buffer ${envRT.width}x${envRT.height} (per eye ${eye.z}x${eye.w})`
				+ ` -> volume buffer ${rt.width}x${rt.height} @ ${(rtScale * 100).toFixed(0)}%`
				+ ` -> framebuffer ${renderer.xr.getSession()?.renderState?.baseLayer?.framebufferWidth ?? '?'}`
				+ `x${renderer.xr.getSession()?.renderState?.baseLayer?.framebufferHeight ?? '?'}`);
		}
	} catch (e) {
		depthNote = T.depthError(e.message);
		depthQuad.visible = false;
		setUseDepth(false);
	}
}

// One control walks the whole cycle: off, each raster in turn, off again.
function stepDepthView() {
	if (!depthKey) { depthKey = true; depthStage = 0; }
	else if (depthStage < DEPTH_STAGES.length - 1) depthStage++;
	else { depthKey = false; depthStage = 0; }
	depthUniforms.stage.value = depthStage;
	updateHud();
}

// ---------------------------------------------------------------- depth ruler

// A reticle showing what the depth sensor reads straight ahead. This is the one
// thing in the scene that deliberately follows your head: it is an instrument,
// not part of the scene, and an instrument that stayed behind when you looked
// away would be useless. Off unless ?ruler=1, and only while an inspection
// stage is up.
const rulerCv = document.createElement('canvas');
rulerCv.width = 256;
rulerCv.height = 128;
const rulerTex = new THREE.CanvasTexture(rulerCv);
rulerTex.colorSpace = THREE.SRGBColorSpace;
rulerTex.minFilter = THREE.LinearFilter;
rulerTex.generateMipmaps = false;

const ruler = new THREE.Mesh(
	new THREE.PlaneGeometry(0.16, 0.08),
	new THREE.MeshBasicMaterial({
		map: rulerTex, transparent: true, depthTest: false, depthWrite: false,
	}));
ruler.renderOrder = 150;
ruler.visible = false;
scene.add(ruler);

let rulerShown = -2;

function drawRuler(metres) {
	const g = rulerCv.getContext('2d');
	g.clearRect(0, 0, rulerCv.width, rulerCv.height);
	g.fillStyle = 'rgba(8,8,16,0.72)';
	g.fillRect(0, 34, rulerCv.width, 60);
	// Crosshair, so it is obvious which point is being measured.
	g.strokeStyle = '#7fd47f';
	g.lineWidth = 2;
	g.beginPath();
	g.moveTo(128, 6); g.lineTo(128, 28);
	g.moveTo(128, 100); g.lineTo(128, 122);
	g.stroke();
	g.font = 'bold 40px DejaVu Sans Mono, monospace';
	g.textAlign = 'center';
	g.textBaseline = 'middle';
	g.fillStyle = metres > 0 ? '#e8e8f0' : '#c88';
	g.fillText(metres > 0 ? `${metres.toFixed(2)} m` : T.rulerNone, 128, 64);
	rulerTex.needsUpdate = true;
}

const _rulerFwd = new THREE.Vector3();
const _rulerPos = new THREE.Vector3();

function updateRuler(cam) {
	ruler.visible = P.ruler === 1 && depthView && !depthNote;
	if (!ruler.visible) return;
	// A metre ahead along the true gaze, facing back at the eye.
	cam.getWorldDirection(_rulerFwd);
	_rulerPos.setFromMatrixPosition(cam.matrixWorld);
	ruler.position.copy(_rulerPos).addScaledVector(_rulerFwd, 1.0);
	ruler.lookAt(_rulerPos);
	if (Math.abs(probeMetres - rulerShown) > 0.005) {
		rulerShown = probeMetres;
		drawRuler(probeMetres);
	}
}

// ---------------------------------------------------------------- vehicle menu

const MENU_ROWS = 10;          // visible rows; the list scrolls inside this

// The footer names the keys you actually have in front of you. Telling a
// desktop user about thumbsticks is worse than saying nothing.
const MENU_FOOT_XR = T.menuFootXr;
const MENU_FOOT_FLAT = T.menuFootFlat;

const menuCv = document.createElement('canvas');
menuCv.width = 720;
menuCv.height = 96 + MENU_ROWS * 44 + 56;
const menuTex = new THREE.CanvasTexture(menuCv);
menuTex.colorSpace = THREE.SRGBColorSpace;
menuTex.minFilter = THREE.LinearFilter;
menuTex.generateMipmaps = false;

const menuPlane = new THREE.Mesh(
	new THREE.PlaneGeometry(1.0, 1.0 * menuCv.height / menuCv.width),
	new THREE.MeshBasicMaterial({
		map: menuTex, transparent: true, depthTest: false, depthWrite: false,
	}));
menuPlane.renderOrder = 130;   // above everything, including the cloud
const menuMesh = twoSided(menuPlane);
menuMesh.visible = false;
scene.add(menuMesh);

let devices = [];
let menuOpen = false;
let menuIndex = 0;
let menuTop = 0;

function drawMenu() {
	const g = menuCv.getContext('2d');
	const W = menuCv.width;
	g.clearRect(0, 0, W, menuCv.height);
	g.fillStyle = 'rgba(8,8,16,0.90)';
	g.fillRect(0, 0, W, menuCv.height);

	g.font = 'bold 34px DejaVu Sans Mono, monospace';
	g.textBaseline = 'top';
	g.fillStyle = '#ffd479';
	g.fillText(T.menuTitle, 28, 26);
	g.font = '24px DejaVu Sans Mono, monospace';
	g.fillStyle = '#8a8a98';
	const count = T.menuCount(devices.length);
	g.fillText(count, W - 28 - g.measureText(count).width, 34);

	// Keep the selection inside the window without moving it more than needed.
	menuTop = Math.min(Math.max(menuTop, menuIndex - MENU_ROWS + 1), menuIndex);
	menuTop = Math.max(0, Math.min(menuTop, Math.max(0, devices.length - MENU_ROWS)));

	g.font = '28px DejaVu Sans Mono, monospace';
	for (let r = 0; r < MENU_ROWS; r++) {
		const i = menuTop + r;
		if (i >= devices.length) break;
		const d = devices[i];
		const y = 96 + r * 44;
		if (i === menuIndex) {
			g.fillStyle = 'rgba(255,212,121,0.22)';
			g.fillRect(14, y - 4, W - 28, 40);
			g.fillStyle = '#ffd479';
			g.fillText('>', 24, y);
		}
		g.fillStyle = i === menuIndex ? '#ffffff' : '#c8c8d6';
		g.fillText(`dvc ${String(d.dvc).padStart(4)}`, 60, y);
		const pts = `${d.rows.toLocaleString(T.numLocale)} ${T.pts}`;
		g.fillStyle = i === menuIndex ? '#e8e8f0' : '#8a8a98';
		g.fillText(pts, W - 28 - g.measureText(pts).width, y);
		if (d.dvc === currentDvc) {
			g.fillStyle = '#7fd47f';
			g.fillText('*', 36, y);
		}
	}

	g.font = '22px DejaVu Sans Mono, monospace';
	g.fillStyle = '#8a8a98';
	g.fillText(renderer.xr.isPresenting ? MENU_FOOT_XR : MENU_FOOT_FLAT, 28, menuCv.height - 42);
	menuTex.needsUpdate = true;
}

const _mh = new THREE.Vector3();

function openMenu() {
	if (menuOpen || devices.length === 0) return;
	menuOpen = true;
	const i = devices.findIndex(d => d.dvc === currentDvc);
	menuIndex = i >= 0 ? i : 0;
	menuTop = Math.max(0, menuIndex - Math.floor(MENU_ROWS / 2));

	// Placed once, in front of wherever you are looking, then left alone -- like
	// everything else here, it must not drift while you read it.
	const cam = headCamera();
	_mh.setFromMatrixPosition(cam.matrixWorld);
	if (renderer.xr.isPresenting) {
		// In the headset the gaze is near-level when you flick the stick, and a
		// yaw-only placement keeps the panel upright in front of you.
		horizontalBasis(cam);
		menuMesh.position.copy(_mh).addScaledVector(_fwd, 1.0);
		menuMesh.position.y = _mh.y - 0.08;
	} else {
		// The desktop camera looks down at the cloud, so a yaw-flattened forward
		// would put the menu above the viewport. Use the true view direction.
		cam.getWorldDirection(_fwd);
		menuMesh.position.copy(_mh).addScaledVector(_fwd, 1.0);
	}
	menuMesh.lookAt(_mh);
	menuMesh.visible = true;
	drawMenu();
}

function closeMenu() {
	menuOpen = false;
	menuMesh.visible = false;
}

function menuMove(dir) {
	if (!menuOpen || devices.length === 0) return;
	menuIndex = Math.min(devices.length - 1, Math.max(0, menuIndex + dir));
	drawMenu();
}

function menuSelect() {
	if (!menuOpen) return;
	const d = devices[menuIndex];
	closeMenu();
	if (d && d.dvc !== currentDvc) loadVolume(d.dvc);
}

let carIndex = -1;

// Steps through the list by index rather than by looking up currentDvc, so
// pressing A twice quickly moves two cars instead of re-requesting the same one
// while the first load is still in flight.
function cycleCar(dir) {
	if (devices.length === 0) return;
	if (carIndex < 0) carIndex = Math.max(0, devices.findIndex(d => d.dvc === currentDvc));
	carIndex = (carIndex + dir + devices.length) % devices.length;
	loadVolume(devices[carIndex].dvc);
}

async function fetchDevices() {
	try {
		const r = await fetch('./api/devices.json');
		if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
		devices = await r.json();
		console.log(`devices: ${devices.length} available`);
	} catch (e) {
		console.warn('device list unavailable:', e.message);
	}
}

// ---------------------------------------------------------------- hud

let volInfo = null;
let fps = 0;

function stats() {
	const t = uniforms.threshold.value;
	const cnt = t <= 0 ? null : Math.ceil(Math.expm1(t * (volInfo?.logMax ?? 0)));
	return {
		thr: t <= 0 ? T.thrOff : T.thrValue(t.toFixed(2), cnt),
		res: `${Math.round(rtScale * 100)}%`,
		spv: (steps / (NB * bbDiag)).toFixed(2),
		gain: SQRT3 / bbDiag,
	};
}

function updateHud() {
	if (!volInfo) return;
	const s = stats();
	hud.innerHTML = [
		`<b>dvc=${volInfo.dvc}</b> <span class="dim">${T.hudPoints}</span> ${volInfo.points.toLocaleString(T.numLocale)}`,
		`<span class="dim">${T.hudVoxels}</span> ${volInfo.nonzero.toLocaleString(T.numLocale)}/512000` +
		`  <span class="dim">${T.hudMax}</span> ${volInfo.maxCount}${T.hudMaxUnit}`,
		`<span class="dim">${T.hudThreshold}</span> ${s.thr}` +
		`  <span class="dim">${T.hudSteps}</span> ${steps} <span class="dim">(${s.spv} ${T.hudSamples}` +
		(s.gain > 1.01 ? `, bbox ${s.gain.toFixed(2)}x` : '') + `)</span>`,
		`<span class="dim">${T.hudBuffer}</span> ${s.res} = ${rt.width}x${rt.height}` +
		`  <b>${fps.toFixed(0)} fps</b>  <span class="dim">${T.hudColour}</span>`,
	].join('<br>');

	drawPanel([
		`dvc ${volInfo.dvc}    ${volInfo.points.toLocaleString(T.numLocale)} ${T.pts}`,
		loadingDvc !== null
			? T.panelLoading(loadingDvc)
			: `${fps.toFixed(0)} fps   ${T.panelBuffer} ${s.res}   ${T.panelSteps} ${steps}`,
		depthView
			? (depthNote || T.depthStage(depthStage, DEPTH_STAGES[depthStage]))
			: T.panelThreshold(s.thr),
		depthView && depthMeta && !depthNote
			? `sensor ${depthMeta.size}  env ${envRects[0].z}x${envRects[0].w}`
			: '',
	]);
}

// ---------------------------------------------------------------- controls

// Provisional desktop framing: aim between the cube and its caption so the
// whole assembly (axes, Z label, panel) fits on load.
const controls = new OrbitControls(camera, renderer.domElement);
// cloudPos, not cloud.position: the group is not placed until applyCloudPos().
controls.target.set(cloudPos.x, cloudPos.y + P.size * 0.30, cloudPos.z);
controls.enableDamping = true;
camera.position.set(
	controls.target.x + P.size * 1.6,
	controls.target.y + P.size * 0.55,
	controls.target.z + P.size * 2.0);
controls.update();

function applyMode() {
	const light = uniforms.lightMode.value === 1;
	axesDark.visible = !light;
	axesLight.visible = light;
	labels.dark.forEach(s => (s.visible = !light));
	labels.light.forEach(s => (s.visible = light));
	backdrop.material.color.set(light ? 0xb3b3ff : 0x05050a);
	document.body.style.background = light ? '#b3b3ff' : '#0a0a0f';
	document.body.style.color = light ? '#40404a' : '#a0a0aa';
}

function bump(what, dir) {
	if (what === 'threshold') {
		uniforms.threshold.value = Math.min(0.99, Math.max(0, uniforms.threshold.value + dir * 0.05));
		applyThreshold();
	} else if (what === 'steps') {
		setSteps(dir > 0 ? steps * 2 : steps / 2);
	}
	updateHud();
}

addEventListener('keydown', e => {
	// Any arrow summons the menu; once it is up, Enter/Right accepts and
	// Esc/Left backs out. Every plausible guess should do the obvious thing.
	switch (e.key) {
		case 'ArrowUp':
			e.preventDefault();
			menuOpen ? menuMove(-1) : openMenu();
			return;
		case 'ArrowDown':
			e.preventDefault();
			menuOpen ? menuMove(+1) : openMenu();
			return;
		case 'ArrowRight':
			e.preventDefault();
			menuOpen ? menuSelect() : openMenu();
			return;
		case 'Enter':
			if (menuOpen) { e.preventDefault(); menuSelect(); }
			return;
		case 'ArrowLeft':
		case 'Escape':
			if (menuOpen) { e.preventDefault(); closeMenu(); }
			return;
	}
	if (menuOpen) return;   // modal, as on the controller
	switch (e.key) {
		case '+': case '=': bump('threshold', +1); break;
		case '-': case '_': bump('threshold', -1); break;
		case 'q': case 'Q': bump('steps', +1); break;
		case 'a': case 'A': bump('steps', -1); break;
		case 'n': case 'N': cycleCar(+1); break;        // = right B
		case 'p': case 'P': cycleCar(-1); break;        // = right A
		case 'd': case 'D': stepDepthView(); break;     // = hold right B, A cycles
		case 'm': case 'M': toggleColormap(); break;    // = left thumbstick press
		case 'b': case 'B': toggleBackdrop(); break;    // = right thumbstick press
	}
});

function toggleColormap() {
	uniforms.lightMode.value ^= 1;
	applyMode();
	updateHud();
}

// ---------------------------------------------------------------- moving the cloud

const MOVE_SPEED = 0.9;    // m/s at full stick deflection
const DEADZONE = 0.15;
const _fwd = new THREE.Vector3(), _right = new THREE.Vector3();

function headCamera() {
	return renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
}

// Yaw-relative, so "right" means right from where you are standing rather than
// some fixed world axis you cannot see.
function horizontalBasis(cam) {
	cam.getWorldDirection(_fwd);
	_fwd.y = 0;
	if (_fwd.lengthSq() < 1e-8) _fwd.set(0, 0, -1); else _fwd.normalize();
	_right.set(-_fwd.z, 0, _fwd.x);
}

function moveCloud(stickX, stickY, vertical, dt) {
	const d = MOVE_SPEED * dt;
	if (vertical) {
		cloudPos.y -= stickY * d;              // stick up = up
	} else {
		horizontalBasis(headCamera());
		cloudPos.addScaledVector(_right, stickX * d)
			.addScaledVector(_fwd, -stickY * d);   // stick up = away
	}
	applyCloudPos();
}

// ---------------------------------------------------------------- controller input

// Touch Plus, xr-standard mapping: buttons[1] squeeze ("smaller" trigger),
// buttons[3] thumbstick press, buttons[4] A/X, buttons[5] B/Y;
// axes[2]/axes[3] are the thumbstick (axes[0]/[1] are the legacy touchpad).
const held = new Map();

function thumbstick(gp) {
	const a = gp.axes;
	return a.length >= 4 ? [a[2] || 0, a[3] || 0] : [a[0] || 0, a[1] || 0];
}

const MENU_OPEN = 0.8;     // deflection that counts as a deliberate flick
const NAV_ENTER = 0.6;     // deflection that counts as one navigation step
const NAV_REPEAT = 0.25;   // seconds between repeats while held

let navActive = false, navPhase = 0;

// True on the frame a navigation step should fire: once immediately on crossing
// the threshold, then on a repeat interval while held. Without this, holding the
// stick scrolls the list at the frame rate.
function stickStep(sy, dt) {
	if (Math.abs(sy) < NAV_ENTER) { navActive = false; navPhase = 0; return false; }
	if (!navActive) { navActive = true; navPhase = 0; return true; }
	navPhase += dt;
	if (navPhase >= NAV_REPEAT) { navPhase = 0; return true; }
	return false;
}

function pollControllers(dt) {
	const session = renderer.xr.getSession();
	if (!session) return;
	for (const src of session.inputSources) {
		const gp = src.gamepad;
		if (!gp) continue;
		const left = src.handedness === 'left';
		const pulse = () => gp.hapticActuators?.[0]?.pulse(0.3, 25);
		const pressed = i => {
			const key = `${src.handedness}${i}`;
			const down = !!gp.buttons[i]?.pressed;
			const edge = down && !held.get(key);
			held.set(key, down);
			return edge;
		};

		if (left) {
			const [sx, sy] = thumbstick(gp);
			const squeeze = !!gp.buttons[1]?.pressed;
			if (squeeze) {
				// Squeeze keeps its existing job -- thumbstick moves the volume
				// up and down -- and additionally makes X the depth inspector.
				// It lives here because there is no menu button to put it on:
				// WebXR's oculus-touch-v3 profile exposes only trigger, squeeze,
				// thumbstick, A/X, B/Y and a touch-only thumbrest. The menu and
				// Meta buttons are reserved by the system and never reach us.
				//
				// One button walks the whole cycle, off included, so there is no
				// separate toggle to get out of step with the stage.
				if (Math.abs(sy) > DEADZONE) moveCloud(0, sy, true, dt);
				if (pressed(4)) { stepDepthView(); pulse(); }         // X
			} else {
				if (Math.abs(sx) > DEADZONE || Math.abs(sy) > DEADZONE) {
					moveCloud(Math.abs(sx) > DEADZONE ? sx : 0,
						Math.abs(sy) > DEADZONE ? sy : 0, false, dt);
				}
				if (pressed(5)) { bump('threshold', +1); pulse(); }   // Y
				if (pressed(4)) { bump('threshold', -1); pulse(); }   // X
				if (pressed(3)) { toggleColormap(); pulse(); }
			}
		} else {
			const [, sy] = thumbstick(gp);
			const step = stickStep(sy, dt);
			if (pressed(3)) { toggleBackdrop(); pulse(); }

			if (menuOpen) {
				// Modal: while the menu is up the right hand drives it and must
				// NOT touch the step count.
				if (step) { menuMove(sy < 0 ? -1 : +1); pulse(); }
				if (pressed(4)) { menuSelect(); pulse(); }             // A
				if (pressed(5)) { closeMenu(); pulse(); }              // B
			} else {
				// A deliberate flick, not a nudge, so a stick resting slightly
				// off-centre cannot summon the menu.
				if (step && Math.abs(sy) > MENU_OPEN) { openMenu(); pulse(); }
				// Plain again: B/A change vehicle, squeeze makes them the step
				// count. The index trigger stays unbound -- hand tracking fires
				// it at random.
				const squeeze = !!gp.buttons[1]?.pressed;
				if (pressed(5)) { squeeze ? bump('steps', +1) : cycleCar(+1); pulse(); }   // B
				if (pressed(4)) { squeeze ? bump('steps', -1) : cycleCar(-1); pulse(); }   // A
			}
		}
	}
}

// ---------------------------------------------------------------- xr session

// AR is the primary mode; the backdrop toggle covers the "no passthrough" case
// without needing a second session. VR stays as a fallback for headsets or
// runtimes where immersive-ar is unavailable.
// depth-sensing is OPTIONAL, never required: on a headset without it the
// session starts exactly as before, minus the overlay. The spec requires the
// depthSensing dictionary to accompany the feature name, so both appear or
// neither does.
const arInit = {
	optionalFeatures: ['local-floor', 'dom-overlay'],
	domOverlay: { root: document.body },
};
if (P.depth) {
	arInit.optionalFeatures.push('depth-sensing');
	arInit.depthSensing = {
		usagePreference: ['gpu-optimized'],          // three only wires up this one
		dataFormatPreference: ['luminance-alpha', 'float32'],
	};
}
const arBtn = ARButton.createButton(renderer, arInit);
const vrBtn = VRButton.createButton(renderer);
arBtn.id = 'ar-btn';
vrBtn.id = 'vr-btn';
document.body.append(arBtn, vrBtn);
localiseXrButton(arBtn);
localiseXrButton(vrBtn);

let needsPlacement = false;

renderer.xr.addEventListener('sessionstart', () => {
	try { renderer.xr.setFoveation(1.0); } catch { /* not every runtime */ }
	controls.enabled = false;
	needsPlacement = true;
	langBtn.style.display = 'none';   // dom-overlay would put it in front of you

	// What we actually got, rather than what we asked for -- this is the first
	// thing to read when the depth overlay stays blank.
	const s = renderer.xr.getSession();
	console.log('[xrviz] session:', JSON.stringify({
		features: s?.enabledFeatures ?? 'n/a',
		depthUsage: s?.depthUsage ?? null,
		depthDataFormat: s?.depthDataFormat ?? null,
		depthNear: s?.renderState?.depthNear,
		depthFar: s?.renderState?.depthFar,
	}));
});

renderer.xr.addEventListener('sessionend', () => {
	controls.enabled = true;
	langBtn.style.display = '';
	depthView = depthKey = false;
	depthQuad.visible = false;
	depthBinding = null;   // tied to the session that just ended
	depthExternal = null;
	depthMeta = null;
	setUseDepth(false);

	// three disposes its XR render target here. Our passes save and restore
	// whatever target is current, so without this the next frame captures the
	// dead one and restores into it -- which is a framebuffer with no live
	// attachments, hence "active draw buffers with missing fragment shader
	// outputs" and a blank page.
	renderer.setRenderTarget(null);
	camera.aspect = innerWidth / innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(innerWidth, innerHeight);
});

// Do not trust the reference space origin. Quest may hand back plain `local`
// even when `local-floor` was requested, and there y=0 is the headset at session
// start (~1.6 m up), not the floor -- so a fixed y=1.3 puts the cloud in the
// ceiling. Placing relative to the measured head pose is correct under either.
//
// Deferred to the first frame with a real pose: at sessionstart the camera
// matrices are still identity.
const _head = new THREE.Vector3();

function placeRelativeToHead() {
	const xrCam = renderer.xr.getCamera();
	if (!xrCam || xrCam.cameras.length === 0) return false;
	_head.setFromMatrixPosition(xrCam.matrixWorld);
	if (_head.lengthSq() === 0) return false;      // pose not established yet

	horizontalBasis(xrCam);
	cloudPos.copy(_head).addScaledVector(_fwd, P.dist);
	cloudPos.y = _head.y - P.size * 0.15;          // a little below eye level
	applyCloudPos();

	if (helpMesh) {
		// The reading matter goes behind you, so it never sits in front of the
		// cloud; turn round to read it.
		helpMesh.position.copy(_head).addScaledVector(_fwd, -P.dist * 1.3);
		helpMesh.position.y = _head.y;
		helpMesh.lookAt(_head);
		helpOffset.copy(helpMesh.position).sub(cloudPos);   // it now rides along
	}
	return true;
}

// ---------------------------------------------------------------- resolution controller

// The step count is never touched here. Below ~1 sample/voxel the slices come
// back and the result is unusable, and that threshold does not care how fast
// the hardware is. Pixels are the only thing left to give up, and on a cloud
// this blurry giving them up is nearly free.
let frameAcc = 0, frameN = 0, lastAdapt = 0;

function adaptResolution(dt, now) {
	if (!P.adapt || !P.rt) return;
	// A backgrounded tab, a stalled compositor or a texture upload produces
	// frames of hundreds of ms that say nothing about how expensive the volume
	// is. Counting them makes the controller chase its own tail down to minimum
	// resolution and stay there.
	if (document.hidden || dt > 100) { lastAdapt = now; frameAcc = 0; frameN = 0; return; }
	frameAcc = Math.max(frameAcc, dt);   // worst frame in the window, not the mean
	frameN++;
	if (now - lastAdapt < 500) return;
	const worst = frameAcc;
	frameAcc = 0; frameN = 0; lastAdapt = now;

	const hz = renderer.xr.isPresenting
		? (renderer.xr.getSession()?.frameRate || 72)
		: 60;
	const budget = 1000 / hz;
	const before = rtScale;

	// Frames are paced by vsync, so while we are keeping up the frame time sits
	// AT the budget, never below it. The old rule only raised the scale when the
	// average fell under 80% of budget -- which vsync makes impossible, so it
	// could go down and never come back up.
	//
	// Instead: probe upward whenever no frame was missed, and back off hard when
	// one was. It settles just under the point where frames start dropping.
	if (worst > budget * 1.4) rtScale = Math.max(0.30, rtScale - 0.06);
	else rtScale = Math.min(1.0, rtScale + 0.02);
	if (rtScale !== before) updateHud();
}

// ---------------------------------------------------------------- diagnostics

// ?debug=1 prints the XR state once a second. Everything here is something that
// has already been wrong at least once, or that differs between the emulator
// and the headset.
let debugDue = 0;

function debugDump(now) {
	if (now < debugDue) return;
	debugDue = now + 1000;
	const xr = renderer.xr;
	const session = xr.getSession();
	const rtNow = renderer.getRenderTarget();
	const o = {
		presenting: xr.isPresenting,
		// null here during an XR frame means the headset framebuffer is not
		// bound and the frame is going to the page instead.
		boundRT: rtNow === null ? 'null (CANVAS)' : `${rtNow.width}x${rtNow.height}`,
		refSpace: xr.getReferenceSpaceType?.() ?? 'n/a',
		frameRate: session?.frameRate ?? null,
		fps: +fps.toFixed(1),
		steps,
		rtScale: +rtScale.toFixed(2),
		rtSize: `${rt.width}x${rt.height}`,
		foveation: xr.getFoveation?.() ?? null,
	};
	if (xr.isPresenting) {
		const xrCam = xr.getCamera();
		o.eyes = xrCam.cameras.map(c => {
			const p = new THREE.Vector3().setFromMatrixPosition(c.matrixWorld);
			return {
				pos: [p.x, p.y, p.z].map(v => +v.toFixed(3)),
				viewport: [c.viewport.x, c.viewport.y, c.viewport.z, c.viewport.w],
			};
		});
		const l = new THREE.Vector3().setFromMatrixPosition(xrCam.matrixWorld);
		o.head = [l.x, l.y, l.z].map(v => +v.toFixed(3));
		const bl = session?.renderState?.baseLayer;
		if (bl) o.baseLayer = `${bl.framebufferWidth}x${bl.framebufferHeight}` +
			(bl.framebuffer ? '' : ' (framebuffer=null -> emulator/default FB)');
	}
	const c = new THREE.Vector3();
	volumeMesh.getWorldPosition(c);
	o.cloud = [c.x, c.y, c.z].map(v => +v.toFixed(3));
	console.log('[xrviz]', JSON.stringify(o));
}

// ---------------------------------------------------------------- loop

let last = performance.now();
let hudDue = 0;

renderer.setAnimationLoop(() => {
	const now = performance.now();
	const dt = now - last;
	last = now;
	// Measured honestly, including slow frames -- this is the indicator, and it
	// should say what is actually happening. Only the resolution controller
	// filters outliers.
	const inst = 1000 / Math.max(dt, 0.5);
	fps = fps ? fps * 0.9 + inst * 0.1 : inst;

	if (needsPlacement && placeRelativeToHead()) needsPlacement = false;
	pollControllers(dt / 1000);
	adaptResolution(dt, now);
	if (controls.enabled) controls.update();
	facePanel(renderer.xr.isPresenting ? renderer.xr.getCamera() : camera);

	depthView = depthKey;
	updateDepth();                        // the banner needs it regardless
	if (depthView) probeDepth(now);
	updateRuler(renderer.xr.isPresenting ? renderer.xr.getCamera() : camera);
	if (P.debug) debugDump(now);
	if (uniforms.useDepth.value === 1) renderEnvPass();   // must precede the volume
	if (P.rt && uniforms.volume.value) renderVolumePass();
	renderer.render(scene, camera);

	if (now > hudDue) { hudDue = now + 250; updateHud(); }
});

addEventListener('resize', () => {
	// The browser fires this while the headset is still presenting -- the DOM
	// overlay and the canvas both change. three refuses to resize then and warns
	// about it; the size is restored on sessionend anyway.
	if (renderer.xr.isPresenting) return;
	camera.aspect = innerWidth / innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- loading

// Fetch and preprocessing happen in a worker so the current cloud keeps
// rendering at full rate while the next one is prepared. If workers are
// unavailable the same code runs inline -- one hitch, but it still works.
let worker = null;
try {
	worker = new Worker(new URL('./volume-worker.js', import.meta.url), { type: 'module' });
} catch (e) {
	console.warn('worker unavailable, loading on the main thread:', e.message);
}

let reqSeq = 0;
const pending = new Map();

if (worker) {
	worker.onmessage = ({ data: { id, ok, info, boxes, buf, error } }) => {
		const p = pending.get(id);
		if (!p) return;
		pending.delete(id);
		ok ? p.resolve({ info, boxes, buf }) : p.reject(new Error(error));
	};
	worker.onerror = e => console.error('volume worker:', e.message);
}

// The device id lives in the path, not the query, so a static export can answer
// this with a file. `limit` is a debugging knob only the live backend can
// honour, so it is only appended when it is actually set.
function fetchVolume(dvc, limit) {
	const url = `./api/volume/${dvc}.vol` + (limit > 0 ? `?limit=${limit}` : '');
	if (!worker) return loadVolumeData(url);
	const id = ++reqSeq;
	return new Promise((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker.postMessage({ id, url });
	});
}

function makeVolumeTexture(data, info) {
	const tex = new THREE.Data3DTexture(data, info.nx, info.ny, info.nz);
	tex.format = THREE.RedFormat;
	tex.type = THREE.UnsignedByteType;
	tex.minFilter = tex.magFilter = THREE.LinearFilter;
	tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
	tex.unpackAlignment = 1;
	tex.needsUpdate = true;
	return tex;
}

// The swap itself: everything is already decoded, so this is one texture upload
// and a few assignments -- a single frame, not a visible gap.
function applyVolume(info, boxes, buf) {
	const data = new Uint8Array(buf, HEADER, info.nx * info.ny * info.nz);
	const tex = uniforms.volume.value;
	if (tex && tex.image.width === info.nx && tex.image.height === info.ny
		&& tex.image.depth === info.nz) {
		// Same dimensions every time, so reuse the GPU allocation rather than
		// churning a new 3D texture per vehicle.
		tex.image.data.set(data);
		tex.needsUpdate = true;
	} else {
		tex?.dispose();
		uniforms.volume.value = makeVolumeTexture(data, info);
	}
	volInfo = info;
	thresholdBoxes = boxes;
	applyThreshold();
	updateHud();
}

let currentDvc = P.dvc;
let loadingDvc = null;
let loadSeq = 0;

async function loadVolume(dvc, { first = false } = {}) {
	if (dvc === loadingDvc) return;
	const token = ++loadSeq;
	loadingDvc = dvc;
	if (first) say(T.statusLoading(dvc));
	updateHud();

	const t0 = performance.now();
	try {
		const { info, boxes, buf } = await fetchVolume(dvc, P.limit);
		// A later pick superseded this one while it was in flight.
		if (token !== loadSeq) return;
		applyVolume(info, boxes, buf);
		currentDvc = dvc;
		const at = devices.findIndex(d => d.dvc === dvc);
		if (at >= 0) carIndex = at;
		status.className = 'hidden';
		console.log(`dvc=${info.dvc}: ${info.points} pts, max=${info.maxCount}, ` +
			`${Math.round(performance.now() - t0)}ms; bbox diag ${bbDiag.toFixed(3)} ` +
			`(${(SQRT3 / bbDiag).toFixed(2)}x finer than the cube)`);
	} catch (e) {
		if (token !== loadSeq) return;
		// A static export holds whichever vehicles were dumped, which need not
		// include the default one. If the URL pinned nothing, fall back to the
		// head of the listing rather than stopping on an error the viewer
		// cannot act on -- which also makes any -dump-dvc choice self-consistent.
		if (first && !qs.has('dvc')) {
			await devicesReady;
			const alt = devices[0];
			if (alt && alt.dvc !== dvc && token === loadSeq) {
				console.warn(`dvc=${dvc} unavailable, falling back to ${alt.dvc}`);
				return loadVolume(alt.dvc, { first: true });
			}
		}
		say(T.statusFailed(dvc, e.message), true);
		console.error(e);
	} finally {
		if (token === loadSeq) {
			loadingDvc = null;
			updateHud();
		}
	}
}

// ---------------------------------------------------------------- boot

// With the low-res buffer the cube lives in volumeScene and reaches the screen
// through blitQuad; without it, straight into the main scene.
if (P.rt) scene.add(blitQuad); else scene.add(volumeGroup);
applyCloudPos();
applyMode();
buildHelp();
// In the background, so the menu opens instantly and the first volume request
// is not held up behind a full aggregate over pos. loadVolume awaits this only
// if it needs the fallback.
const devicesReady = fetchDevices();
loadVolume(P.dvc, { first: true });
