// Controller rays and the buttons they press.
//
// The controls hang off the caption panel, as children, so they inherit its
// placement and its yaw billboarding for free -- `applyCloudPos()` and
// `facePanel()` keep working exactly as they did and know nothing about this
// file. That also makes the hit test cheap: every control lies in ONE plane, so
// a ray is one plane intersection plus a rectangle lookup, not a scene graph
// traversal. No Raycaster, because `intersectObjects()` allocates a result array
// per pointer per frame and we would throw all of it away.
//
// These controls exist only while a session is presenting; the flat page has
// its own, in HTML, built by page.js. Three kinds of pointer can drive them:
//
//   XR controllers  ray from the target-ray space, pressed with the trigger
//   hand tracking   the same, since a pinch is a `select` on a targetRaySpace
//   mouse / touch   a ray through the cursor, pressed by a CLICK -- a drag of
//                   more than a few pixels is a camera orbit and is ignored,
//                   which is what keeps OrbitControls swipes working untouched
//
// The third is inert in practice, since the controls are not shown outside a
// session. It is kept because it costs nothing when they are hidden, it is what
// the headless tests drive through `force()`, and it is the whole of what a
// phone would need if these ever went on the flat page.
//
// Nothing here runs per-pixel and nothing allocates in the frame loop: one
// matrix inversion, a handful of dot products per pointer, and two extra draw
// calls per control. The volume shader will not notice.

import * as THREE from 'three';

// ---------------------------------------------------------------- layout

// Fractions of the caption panel's own width, so ?size= scales the controls
// with the caption instead of leaving them behind. Square, and about 6.5 cm a
// side at the default size -- roughly 3 degrees at arm's length, which is well
// clear of the half-degree or so a held controller wanders by.
const SQ = 0.075;      // control side
const GAP = 0.022;     // between neighbours
const PAD = 0.030;     // between the row and the panel edge

// `col` counts slots of (side + gap) from the centre of the row, so a row is
// laid out by writing down where each control sits rather than by summing
// widths. `row` is which side of the caption it hangs on, `w` widens a control
// without moving its neighbours.
//
// The menu button is deliberately absent: the vehicle list wants clickable rows
// before a button to summon it is worth anything.
const CONTROLS = [
	{ id: 'car-', glyph: '<', row: 'above', col: -4.5 },
	{ id: 'grab', glyph: '≡', row: 'above', col: 0, w: 2.2, kind: 'handle' },
	{ id: 'car+', glyph: '>', row: 'above', col: +4.5 },
	{ id: 'thr-', glyph: '−', row: 'below', col: -0.5, repeat: true },
	{ id: 'thr+', glyph: '+', row: 'below', col: +0.5, repeat: true },
];

// Beyond this the ray is pointing at something else entirely; a control that
// can be pressed from across the room is a control pressed by accident.
const MAX_REACH = 6.0;

// Held-down repeat. XR only: the flat pointer fires on release, so there is no
// hold to repeat. Threshold moves in steps of 0.05, so crossing the range by
// discrete presses is twenty of them.
const REPEAT_DELAY = 450;
const REPEAT_EVERY = 250;

const FLASH_MS = 140;   // how long the "pressed" colour stays after a press

// Idle is lighter than the banner it sits on and clearly not solid; hover is
// dark blue; the press flash is light blue, and the handle stays at it for as
// long as it is held.
const IDLE = { color: 0x9a9aa8, opacity: 0.42 };
const HANDLE_IDLE = { color: 0x9a9aa8, opacity: 0.55 };   // reads as solid: it is grabbable
const HOVER = { color: 0x1d3f8f, opacity: 0.78 };
const PRESS = { color: 0x7fb6ff, opacity: 0.92 };

// One appearance, always. A ray that dims when it is pointing at nothing is a
// ray you cannot find against passthrough -- and finding it is the whole point
// of drawing it.
const RAY = { color: 0x9fd0ff, opacity: 0.85 };

// The beam starts a hand's breadth in front of the controller rather than
// inside it: a line emerging from the middle of the thing you are holding reads
// as a mistake. Only the drawing moves -- the ray that is tested still starts at
// the target-ray origin, where the runtime says it does.
const RAY_START = 0.10;
const RAY_FREE = 1.2;     // how long it is drawn when it has hit nothing

// The background shader writes its colour to the framebuffer untouched -- three
// only applies the working-space -> output conversion inside its own materials,
// and this one is hand-written, like the volume's. So the uniform has to hold
// the sRGB value as written above rather than the linearised one THREE.Color
// produces by default, or every control comes out about half as bright as the
// hex says. The ray is a LineBasicMaterial, which does go through three's
// conversion, so there the plain setHex is the correct one.
const asWritten = hex => new THREE.Color().setHex(hex, THREE.LinearSRGBColorSpace);

// ---------------------------------------------------------------- shaders

// The control's plate as a signed distance field rather than a texture: an
// antialiased edge at any distance without mipmaps, nothing to repaint when a
// colour changes, and the shape follows the geometry instead of being baked at
// one aspect ratio. Colour and opacity are uniforms, so a state change is two
// assignments and zero uploads. `radius` is 0 for the square plates we settled
// on; the term is left in because rounding is one uniform away.
const BTN_VERT = /* glsl */`
out vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const BTN_FRAG = /* glsl */`
precision highp float;
uniform vec3  color;
uniform float opacity;
uniform vec2  halfSize;   // metres
uniform float radius;     // metres
in vec2 vUv;
out vec4 fragColor;

void main() {
	vec2 p = (vUv - 0.5) * 2.0 * halfSize;
	vec2 q = abs(p) - halfSize + radius;
	float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
	// One pixel of feather, measured in the same units as d, so the edge is
	// equally soft whether the control is at arm's length or across the room.
	float aa = max(fwidth(d), 1e-5);
	float a = opacity * (1.0 - smoothstep(-aa, aa, d));
	if (a <= 0.002) discard;
	fragColor = vec4(color, a);
}
`;

// ---------------------------------------------------------------- glyphs

// One small canvas per control, built once. White on transparent, so the glyph
// stays readable over every background colour the control takes.
function texture(cv) {
	const tex = new THREE.CanvasTexture(cv);
	tex.colorSpace = THREE.SRGBColorSpace;
	tex.minFilter = tex.magFilter = THREE.LinearFilter;
	tex.generateMipmaps = false;
	return tex;
}

function glyphTexture(ch) {
	const S = 128;
	const cv = document.createElement('canvas');
	cv.width = cv.height = S;
	const g = cv.getContext('2d');
	g.fillStyle = '#f0f0f6';
	g.font = `bold ${Math.round(S * 0.66)}px DejaVu Sans Mono, monospace`;
	g.textAlign = 'center';
	g.textBaseline = 'middle';
	g.fillText(ch, S / 2, S / 2 + S * 0.03);
	return texture(cv);
}

// The handle gets bars drawn to its own width rather than a character. A glyph
// is as wide as the font made it, which on a plate twice the width of a button
// leaves a small mark adrift in a large rectangle -- and the thing has to read
// as "grab here" at a glance, across a room, in passthrough.
function barsTexture(aspect) {
	const S = 128;
	const cv = document.createElement('canvas');
	cv.height = S;
	cv.width = Math.round(S * aspect);
	const g = cv.getContext('2d');
	g.fillStyle = '#f0f0f6';
	const w = cv.width * 0.84, h = S * 0.13, gap = S * 0.20;
	for (const dy of [-gap / 2 - h / 2, gap / 2 + h / 2]) {
		g.fillRect((cv.width - w) / 2, S / 2 + dy - h / 2, w, h);
	}
	return texture(cv);
}

// ---------------------------------------------------------------- build

export function buildUI({ renderer, scene, camera, panel, actions, drag = null,
	blocked = () => false, visible = () => true, debug = 0 }) {
	const W = panel.geometry.parameters.width;
	const H = panel.geometry.parameters.height;
	const sq = SQ * W, gap = GAP * W, pad = PAD * W;

	// Shared by every control: the quads differ only in their transform.
	const quad = new THREE.PlaneGeometry(1, 1);

	const controls = CONTROLS.map(c => {
		const bw = sq * (c.w ?? 1), bh = sq;
		const cx = c.col * (sq + gap);
		const cy = (c.row === 'above' ? 1 : -1) * (H / 2 + bh / 2 + pad);
		const idle = c.kind === 'handle' ? HANDLE_IDLE : IDLE;

		const bg = new THREE.Mesh(quad, new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			uniforms: {
				color: { value: asWritten(idle.color) },
				opacity: { value: idle.opacity },
				halfSize: { value: new THREE.Vector2(bw / 2, bh / 2) },
				radius: { value: 0 },
			},
			vertexShader: BTN_VERT,
			fragmentShader: BTN_FRAG,
			transparent: true,
			depthTest: false,
			depthWrite: false,
		}));
		bg.scale.set(bw, bh, 1);
		// In front of the caption (120) and of the volume composite (100), under
		// the vehicle menu (130). Everything here has depthTest off, so
		// renderOrder is the only thing deciding what covers what.
		bg.renderOrder = 124;

		const handle = c.kind === 'handle';
		const gw = handle ? bw * 0.72 : Math.min(bw, bh) * 0.82;
		const gh = handle ? bh * 0.72 : gw;
		const glyph = new THREE.Mesh(quad, new THREE.MeshBasicMaterial({
			map: handle ? barsTexture(gw / gh) : glyphTexture(c.glyph),
			transparent: true,
			depthTest: false,
			depthWrite: false,
		}));
		glyph.scale.set(gw, gh, 1);
		glyph.renderOrder = 125;

		const group = new THREE.Group();
		// A whisker in front of the caption quad. Nothing depth-tests, but the
		// panel is a sibling plane at z=0 and coincident geometry confuses
		// three's own transparent sort.
		group.position.set(cx, cy, 0.002);
		glyph.position.z = 0.001;
		group.add(bg, glyph);

		return {
			id: c.id,
			repeat: !!c.repeat,
			handle,
			idle,
			cx, cy,
			hw: bw / 2, hh: bh / 2,
			mat: bg.material,
			hover: 0,            // pointers resting on it this frame
			grabbed: false,      // the handle, while someone is dragging by it
			flashUntil: 0,
			state: '',
			group,
		};
	});

	const uiGroup = new THREE.Group();
	uiGroup.add(...controls.map(c => c.group));
	panel.add(uiGroup);

	// ------------------------------------------------------------ pointers

	const pointers = [];

	// Both controller slots have to exist BEFORE a session starts: three only
	// hands an input source to a controller that is already in its array
	// (WebXRManager.onInputSourcesChange), so a controller created later never
	// receives one.
	for (let i = 0; i < 2; i++) {
		const obj = renderer.xr.getController(i);

		const line = new THREE.Line(
			new THREE.BufferGeometry().setFromPoints(
				[new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
			new THREE.LineBasicMaterial({
				color: RAY.color,
				transparent: true,
				opacity: RAY.opacity,
				depthTest: false,
				depthWrite: false,
			}));
		line.renderOrder = 128;
		line.frustumCulled = false;   // a unit-long line, scaled; the bounds lie
		// The gap is made by moving the whole line forward and drawing only what
		// is left of the distance, so the far end still lands exactly on the hit.
		line.position.z = -RAY_START;
		obj.add(line);
		scene.add(obj);

		const p = {
			kind: 'xr', obj, line, src: null,
			down: false, downEdge: false, holding: null, repeatAt: 0, target: null,
			dragging: null,                  // the control being dragged by
			grab: new THREE.Vector3(),       // what is being dragged, in this controller's space
		};

		obj.addEventListener('connected', e => {
			p.src = e.data;
			// On a phone the "controller" is the touch point: a transient input
			// source with targetRayMode 'screen'. The ray is real and the hit
			// test works, but drawing a beam out of the screen is nonsense.
			line.visible = e.data.targetRayMode !== 'screen';
		});
		obj.addEventListener('disconnected', () => {
			p.src = null;
			p.down = p.downEdge = false;
			p.holding = null;
			stopDrag(p);
		});
		obj.addEventListener('selectstart', () => { p.down = true; p.downEdge = true; });
		obj.addEventListener('selectend', () => { p.down = false; });

		pointers.push(p);
	}

	// The flat pointer. Deliberately fires on RELEASE and only when the pointer
	// barely moved: pointerdown belongs to OrbitControls, and stealing it would
	// break orbiting with the mouse and swiping on a phone -- which is the one
	// thing that must keep working exactly as before. Nothing here calls
	// preventDefault or stopPropagation, so OrbitControls sees every event.
	const flat = { kind: 'flat', ndc: null, down: false, downEdge: false, holding: null, repeatAt: 0, target: null };
	pointers.push(flat);

	const DRAG_SLOP = 8;     // px; beyond this the gesture was a camera move
	let downX = 0, downY = 0, downId = -1;

	const toNdc = e => {
		const r = renderer.domElement.getBoundingClientRect();
		if (!r.width || !r.height) return null;
		if (!flat.ndc) flat.ndc = new THREE.Vector2();
		return flat.ndc.set(
			((e.clientX - r.left) / r.width) * 2 - 1,
			-(((e.clientY - r.top) / r.height) * 2 - 1));
	};

	const el = renderer.domElement;
	el.addEventListener('pointermove', e => {
		if (renderer.xr.isPresenting) return;
		toNdc(e);
	}, { passive: true });

	el.addEventListener('pointerdown', e => {
		if (renderer.xr.isPresenting) return;
		downId = e.pointerId;
		downX = e.clientX;
		downY = e.clientY;
		toNdc(e);
	}, { passive: true });

	el.addEventListener('pointerup', e => {
		if (renderer.xr.isPresenting || e.pointerId !== downId) return;
		downId = -1;
		const moved = Math.abs(e.clientX - downX) > DRAG_SLOP
			|| Math.abs(e.clientY - downY) > DRAG_SLOP;
		toNdc(e);
		if (!moved) flat.downEdge = true;
		// A finger leaves nothing behind. Without this the last control touched
		// stays lit until something else is touched.
		if (e.pointerType !== 'mouse') flat.ndc = null;
	}, { passive: true });

	el.addEventListener('pointerleave', () => { flat.ndc = null; }, { passive: true });

	// ------------------------------------------------------------ hit test

	const _inv = new THREE.Matrix4();
	const _o = new THREE.Vector3();
	const _d = new THREE.Vector3();
	const _ro = new THREE.Vector3();
	const _rd = new THREE.Vector3();
	let hitDist = 0;

	// Ray (world space) against the plane every control lies in, then which
	// rectangle the hit landed in. `_inv` is the panel's inverse world matrix,
	// refreshed once per frame.
	function pick(origin, dir) {
		_o.copy(origin).applyMatrix4(_inv);
		_d.copy(dir).transformDirection(_inv);
		if (Math.abs(_d.z) < 1e-6) return null;          // ray parallel to the panel
		const t = -_o.z / _d.z;
		if (t <= 0 || t > MAX_REACH) return null;
		hitDist = t;
		const x = _o.x + _d.x * t;
		const y = _o.y + _d.y * t;
		for (const c of controls) {
			if (Math.abs(x - c.cx) <= c.hw && Math.abs(y - c.cy) <= c.hh) return c;
		}
		return null;
	}

	function pulse(p, strength, ms) {
		if (p.kind === 'xr') p.src?.gamepad?.hapticActuators?.[0]?.pulse(strength, ms);
	}

	function fire(c, p, now) {
		c.flashUntil = now + FLASH_MS;
		pulse(p, 0.35, 22);
		actions[c.id]?.();
	}

	function aimXr(p) {
		if (!p.src || !p.obj.visible) return null;
		p.obj.updateMatrixWorld();
		_ro.setFromMatrixPosition(p.obj.matrixWorld);
		_rd.set(0, 0, -1).transformDirection(p.obj.matrixWorld);
		return pick(_ro, _rd);
	}

	function aimFlat() {
		if (!flat.ndc || renderer.xr.isPresenting) return null;
		camera.updateMatrixWorld();
		_ro.setFromMatrixPosition(camera.matrixWorld);
		_rd.set(flat.ndc.x, flat.ndc.y, 0.5).unproject(camera).sub(_ro).normalize();
		return pick(_ro, _rd);
	}

	function release(p) {
		p.down = p.downEdge = false;
		p.holding = null;
		p.target = null;
		stopDrag(p);
	}

	// ------------------------------------------------------------ the handle

	// A rigid grab, the way every VR application does it: what is held is stored
	// in the CONTROLLER's frame at the moment of grabbing, and re-expressed in
	// world space every frame afterwards. Two things fall out of that for free.
	// Walking carries the cloud along, because the controller pose is in the
	// same world space the cloud lives in and walking moves it. And turning the
	// wrist swings the cloud through an arc, so it can be pushed away and pulled
	// back without a separate control for distance.
	//
	// Only the POSITION is written back. The cloud's orientation is load-bearing
	// -- the axes mean something -- and a cloud that rolled with your wrist would
	// be both wrong and, per the comfort rules, unpleasant.
	const _grab = new THREE.Vector3();
	let dragCount = 0;

	function startDrag(c, p) {
		if (!drag || dragCount > 0) return;    // one hand at a time; two would fight
		drag.get(_grab);                       // where it is now, in world space
		p.obj.updateMatrixWorld();
		p.obj.worldToLocal(_grab);             // ... and in the controller's frame
		p.grab.copy(_grab);
		p.dragging = c;
		c.grabbed = true;
		dragCount++;
		drag.active(true);
		pulse(p, 0.5, 30);
	}

	function moveDrag(p) {
		_grab.copy(p.grab).applyMatrix4(p.obj.matrixWorld);
		drag.set(_grab);
	}

	function stopDrag(p) {
		if (!p.dragging) return;
		p.dragging.grabbed = false;
		p.dragging = null;
		dragCount = Math.max(0, dragCount - 1);
		if (dragCount === 0) drag.active(false);
		pulse(p, 0.2, 12);
	}

	// ------------------------------------------------------------ frame

	function paint(c, now) {
		// The handle stays lit for as long as it is held; a button flashes and
		// lets go, so a press reads as an event rather than as a state.
		const s = (c.grabbed || now < c.flashUntil) ? 'press' : (c.hover > 0 ? 'hover' : 'idle');
		if (s === c.state) return;                   // only touch uniforms on a change
		c.state = s;
		const v = s === 'press' ? PRESS : (s === 'hover' ? HOVER : c.idle);
		c.mat.uniforms.color.value.setHex(v.color, THREE.LinearSRGBColorSpace);
		c.mat.uniforms.opacity.value = v.opacity;
	}

	// Only the test harness sets this, to drive the in-scene controls on a flat
	// page where `visible()` is false because no session is running.
	let forced = false;

	function update() {
		const now = performance.now();

		// Away in two cases: the vehicle menu is modal, as it is for the sticks
		// and the keyboard, and there is no session, where the HTML controls
		// have the job instead.
		if (blocked() || !(visible() || forced)) {
			if (uiGroup.visible) {
				uiGroup.visible = false;
				for (const p of pointers) release(p);
				for (const c of controls) { c.hover = 0; c.flashUntil = 0; paint(c, now); }
			}
			for (const p of pointers) if (p.line) p.line.visible = false;
			return;
		}
		uiGroup.visible = true;

		// facePanel() has just re-yawed the panel and the render that would
		// refresh its world matrix has not happened yet, so this is where the
		// controls' placement becomes current -- for the hit test and for the
		// draw that follows.
		panel.updateMatrixWorld();
		_inv.copy(panel.matrixWorld).invert();

		for (const c of controls) c.hover = 0;

		for (const p of pointers) {
			const target = p.kind === 'xr' ? aimXr(p) : aimFlat();

			if (p.line) {
				const lit = p.src && p.src.targetRayMode !== 'screen' && p.obj.visible;
				p.line.visible = !!lit;
				// Stops at what it hit, otherwise a fixed arm's length. The
				// colour never changes: the control lighting up is the feedback.
				if (lit) {
					const reach = (target || p.dragging) ? hitDist : RAY_FREE;
					p.line.scale.z = Math.max(0.02, reach - RAY_START);
				}
			}

			// A light tick when the ray arrives on a control, so the hand knows
			// it is on target without looking for the colour change.
			if (target && target !== p.target) pulse(p, 0.12, 8);
			p.target = target;

			if (target) target.hover++;

			// The drag runs off the held state alone, never off the hit test:
			// once grabbed, the cloud follows the hand until the trigger is let
			// go, wherever the ray wanders. (It does not wander -- the panel is
			// being carried by the same hand -- but a grab that can be lost by
			// aiming badly is a grab that drops things.)
			if (p.dragging) {
				if (p.down) moveDrag(p);
				else stopDrag(p);
			}

			if (p.downEdge) {
				p.downEdge = false;
				if (target?.handle) {
					// Only a pointer with a pose can carry something; the flat
					// pointer has already been released by the time it fires.
					if (p.kind === 'xr' && p.down) startDrag(target, p);
				} else if (target) {
					fire(target, p, now);
					// A tap shorter than one frame arrives as press+release
					// together: fire it, but do not enter the held state.
					p.holding = p.down ? target : null;
					p.repeatAt = now + REPEAT_DELAY;
				}
			} else if (p.holding) {
				if (!p.down || target !== p.holding) p.holding = null;
				else if (p.holding.repeat && now >= p.repeatAt) {
					fire(p.holding, p, now);
					p.repeatAt = now + REPEAT_EVERY;
				}
			}
		}

		for (const c of controls) paint(c, now);
	}

	// Called when a session ends: the controllers are gone and any press that
	// was in flight went with them.
	function reset() {
		for (const p of pointers) release(p);
		const now = performance.now();
		for (const c of controls) { c.hover = 0; c.flashUntil = 0; paint(c, now); }
	}

	// ?debug=1 opens the UI to a test harness: where each control landed in page
	// pixels and what state it is in, plus the pointers themselves. Coordinates
	// typed into a test by hand go stale the moment the layout moves, and the
	// XR half of this file is otherwise only reachable by putting a headset on
	// -- a fake pointer aimed at a control exercises the same ray, the same hit
	// test and the same press path.
	if (debug) {
		const _p = new THREE.Vector3();
		window.__xrvizUI = {
			pointers,
			panel,
			force: on => { forced = on; },
			controls: () => controls.map(c => {
				_p.set(c.cx, c.cy, 0).applyMatrix4(panel.matrixWorld).project(camera);
				return {
					id: c.id,
					state: c.state,
					x: Math.round((_p.x * 0.5 + 0.5) * innerWidth),
					y: Math.round((-_p.y * 0.5 + 0.5) * innerHeight),
					cx: c.cx, cy: c.cy,
				};
			}),
		};
	}

	return { update, reset };
}
