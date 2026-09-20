// The flat page: which device is looking at it, and the controls drawn with
// HTML rather than in the scene.
//
// Two layouts, three names. A headset's browser window and a desktop window
// both have room for the instructions, the readout and a labelled button; a
// phone has room for none of it and gets bare glyphs at the two edges. The
// third name exists anyway, because which XR button to offer is a separate
// question from how much room there is, and because ?platform= has to be able
// to name what it is overriding.
//
// Nothing here runs per frame. It is built once, and afterwards it is a handful
// of click handlers calling the same functions the keys call.

import { ARButton } from 'three/addons/ARButton.js';
import { VRButton } from 'three/addons/VRButton.js';
import { T, OTHER_LANG, setLang, localiseXrButton } from './i18n.js';

const KINDS = ['desktop', 'mobile', 'headset'];

// A headset browser is the one case worth reading the user agent for: it is a
// small fixed set of browsers, and the honest signal -- immersive-vr support --
// only arrives asynchronously, after the page has already had to lay itself
// out. The async answer then corrects the guess below.
const HEADSET_UA = /OculusBrowser|Quest|Pico|Wolvic|VisionOS/i;

// "Mobile" is in the UA of every phone browser worth naming, and Chrome's
// userAgentData answers the question outright. An Android tablet says neither,
// and lands on the roomy layout, which is the right answer for a tablet.
const PHONE_UA = /Android.*Mobile|Mobile.*Android|iPhone|iPod|IEMobile|BlackBerry|Opera Mini/i;

// Below either of these the instructions and the readout start landing on top
// of each other: the readout alone is about 520 px of monospace and the
// instructions about 400 px, so they collide long before a phone's width.
const ROOM_W = 900;
const ROOM_H = 520;

function mm(q) {
	try { return matchMedia(q).matches; } catch { return null; }
}

// Everything the decision looks at, and what it made of it. One place, logged
// in full, because the only way to find out what a real phone reports is to
// read it off a real phone.
export function deviceSignals() {
	const ua = navigator.userAgent;
	const d = navigator.userAgentData;
	return {
		ua,
		uaDataMobile: d ? d.mobile : null,
		uaDataPlatform: d ? d.platform : null,
		maxTouchPoints: navigator.maxTouchPoints,
		inner: `${innerWidth}x${innerHeight}`,
		screen: `${screen.width}x${screen.height}`,
		visualViewport: self.visualViewport
			? `${Math.round(visualViewport.width)}x${Math.round(visualViewport.height)}`
			: null,
		devicePixelRatio,
		orientation: screen.orientation ? screen.orientation.type : null,
		'pointer:coarse': mm('(pointer: coarse)'),
		'pointer:fine': mm('(pointer: fine)'),
		'any-pointer:coarse': mm('(any-pointer: coarse)'),
		'hover:none': mm('(hover: none)'),
		'hover:hover': mm('(hover: hover)'),
		headsetUA: HEADSET_UA.test(ua),
		phoneUA: PHONE_UA.test(ua),
		touch: navigator.maxTouchPoints > 0 || mm('(pointer: coarse)') === true,
		roomy: innerWidth >= ROOM_W && innerHeight >= ROOM_H,
	};
}

export function detectPlatform(override) {
	const s = deviceSignals();
	let kind, why;

	if (KINDS.includes(override)) {
		kind = override;
		why = `?platform=${override}`;
	} else if (s.headsetUA) {
		kind = 'headset';
		why = 'headset user agent';
	} else if (s.uaDataMobile === true || s.phoneUA) {
		// The device says it is a phone. Believe it, whatever size it reports:
		// "request desktop site" widens the viewport without changing what the
		// thing is.
		kind = 'mobile';
		why = s.uaDataMobile === true ? 'userAgentData.mobile' : 'phone user agent';
	} else if (s.touch && !s.roomy) {
		// No name to go on, but it is touch-driven and there is no room. An
		// earlier version also demanded (hover: none) and a viewport under
		// 500 px, and a real phone failed both.
		kind = 'mobile';
		why = 'touch and no room';
	} else {
		kind = 'desktop';
		why = 'nothing said otherwise';
	}

	console.log(`[xrviz] device -> ${kind} (${why})`);
	console.log('[xrviz] device signals: ' + JSON.stringify(s, null, 1));
	return kind;
}

// Whether there is room for the reading matter, which is a different question
// from what the device is: a desktop window dragged narrow has a phone's
// problem, and a phone whose identity we failed to read still has to be
// legible. Measured, so it cannot be got wrong by a user-agent string, and
// re-measured when the window changes.
export function isCompact(platform) {
	return platform === 'mobile' || innerWidth < ROOM_W || innerHeight < ROOM_H;
}

// A phone can offer immersive-ar; it never offers immersive-vr. So a device
// that was taken for a phone and then turns out to support VR is a headset
// whose browser this file has not heard of. A desktop with a tethered headset
// also answers yes, and is left alone: it is on the roomy layout already, which
// is the right one for it.
function reconcile(platform, vr) {
	return (vr && platform === 'mobile') ? 'headset' : platform;
}

function button(cls, glyph, word, onClick) {
	const b = document.createElement('button');
	b.type = 'button';
	b.className = cls;
	const g = document.createElement('span');
	g.className = 'g';
	g.textContent = glyph;
	b.append(g);
	if (word) {
		const w = document.createElement('span');
		w.className = 'w';
		w.textContent = word;
		b.append(w);
	}
	b.addEventListener('click', onClick);
	return b;
}

export function buildPage({ renderer, arInit, actions, platform, onLayout }) {
	const root = document.documentElement;
	let kind = platform;
	let compact = null;

	// Two attributes, two questions. `platform` is what the device is, and only
	// the XR button and the menu button care. `compact` is whether there is room,
	// and it is what the layout is actually written against.
	const layout = () => {
		root.dataset.platform = kind;
		const now = isCompact(kind);
		if (now === compact) return;
		compact = now;
		root.dataset.compact = compact ? '1' : '0';
		console.log(`[xrviz] layout -> ${compact ? 'compact' : 'roomy'}`
			+ ` (platform=${kind}, ${innerWidth}x${innerHeight})`);
		if (onLayout) onLayout(compact);
	};
	layout();
	const setPlatform = p => { kind = p; layout(); };

	// A phone turned sideways, or a window dragged narrow, changes the answer.
	addEventListener('resize', layout, { passive: true });

	// Left: the threshold, plus above minus. Right: the vehicle, and the menu
	// the arrow keys already open. Both are the same actions the keyboard and
	// the controllers reach, never a second implementation of anything.
	document.getElementById('ctl-left').append(
		button('ctl-btn', '+', null, () => actions['thr+']?.()),
		button('ctl-btn', '−', null, () => actions['thr-']?.()));

	document.getElementById('ctl-right').append(
		button('ctl-btn', '<', T.btnPrev, () => actions['car-']?.()),
		button('ctl-btn', '>', T.btnNext, () => actions['car+']?.()),
		// Omitted on a phone: it costs a whole button's worth of screen to open
		// a list that then has nothing to drive it.
		button('ctl-btn menu-btn', '≡', T.btnMenu, () => actions['menu']?.()));

	// The language toggle joins the XR buttons in one row, rather than being
	// placed by hand next to them. Their labels change width -- "START AR" and
	// "AR NOT SUPPORTED" are not the same size -- and three sets that width
	// inline, from an async check, which is what used to make the three of them
	// overlap. A flex row cannot overlap whatever it is given.
	const bar = document.getElementById('xr-bar');
	const langBtn = button('ctl-btn lang-btn', T.langFlag, T.langLabel,
		() => setLang(OTHER_LANG));
	langBtn.id = 'lang-btn';
	bar.append(langBtn);

	// Asked before anything is built, so an unsupported mode produces no button
	// at all rather than a wide "NOT SUPPORTED" one. Both unsupported: one
	// caption, which is the whole truth in the space of half a button.
	(async () => {
		const can = async mode => {
			try { return !!(navigator.xr && await navigator.xr.isSessionSupported(mode)); }
			catch { return false; }
		};
		const [ar, vr] = await Promise.all([can('immersive-ar'), can('immersive-vr')]);
		const settled = reconcile(kind, vr);
		console.log(`[xrviz] webxr: immersive-ar=${ar} immersive-vr=${vr}`
			+ ` -> platform ${kind}${settled === kind ? '' : ` -> ${settled}`}`);
		setPlatform(settled);

		if (!ar && !vr) {
			const note = document.createElement('div');
			note.id = 'xr-none';
			note.textContent = T.xrUnavailable;
			bar.insertBefore(note, langBtn);
			return;
		}
		if (ar) {
			const b = ARButton.createButton(renderer, arInit);
			b.id = 'ar-btn';
			bar.insertBefore(b, langBtn);
			localiseXrButton(b);
		}
		if (vr) {
			const b = VRButton.createButton(renderer);
			b.id = 'vr-btn';
			bar.insertBefore(b, langBtn);
			localiseXrButton(b);
		}
	})();

	return {
		// One class, and the stylesheet decides what a session hides. The
		// instructions and the readout stay: in AR they are the dom-overlay, and
		// the in-scene controls have taken over from the HTML ones.
		presenting: on => document.body.classList.toggle('presenting', on),
	};
}
