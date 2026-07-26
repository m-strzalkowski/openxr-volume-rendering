// Localisation. Two languages, picked once at load; changing it reloads the
// page. Almost everything visible here -- canvas textures, the help banner, the
// shaders' companion labels -- is built once at startup from strings, so a live
// switch would mean rebuilding the scene for a button nobody presses twice.
//
// Polish is the default and the system locale is deliberately NOT consulted:
// the headset is shared and configured by whoever used it last, which is no
// basis for choosing the language of the exhibit. ?lang=en (or the button)
// overrides, and the choice sticks.
//
// The toggle exists only on the flat page. In XR there is no keyboard, the
// controller buttons are all spoken for, and a mid-session reload would drop
// the session.

const STORE = 'xrviz.lang';

function initialLang() {
	const q = new URLSearchParams(location.search).get('lang');
	if (q === 'pl' || q === 'en') return q;
	try {
		const s = localStorage.getItem(STORE);
		if (s === 'pl' || s === 'en') return s;
	} catch { /* storage blocked; fall through to the default */ }
	return 'pl';
}

export const LANG = initialLang();
export const OTHER_LANG = LANG === 'pl' ? 'en' : 'pl';

// Written to the URL as well as to storage: a ?lang= left over from before the
// switch would otherwise outrank the new choice on the very next load. It also
// makes the link shareable in a fixed language.
export function setLang(lang) {
	try { localStorage.setItem(STORE, lang); } catch { /* storage blocked */ }
	const url = new URL(location.href);
	url.searchParams.set('lang', lang);
	location.replace(url);
}

const STRINGS = {
	pl: {
		title: 'Gęstość przyspieszeń 3D — XR',
		// The button offers the OTHER language, so it flies the other flag.
		// Regional indicators GB, spelled out rather than pasted, because a pair
		// of them is one glyph that no editor will show you as two code points.
		langOther: '\u{1F1EC}\u{1F1E7} English',
		loadingVolume: 'ładuję wolumen…',
		helpFile: './VR_inputs.PL_pl.txt',
		numLocale: 'pl',

		depthStages: ['źródło (nearest)', 'prepass: metry', 'prepass: przesłanianie'],
		depthArOnly: 'głębia tylko w sesji AR',
		depthUnavailable: 'depth-sensing niedostępne na tym urządzeniu',
		depthNoPose: 'brak pozy',
		depthNoFrame: 'brak klatki głębi',
		depthError: m => `głębia: ${m}`,
		depthStage: (i, name) => `głębia ${i}: ${name}`,
		rulerNone: 'brak',

		menuTitle: 'POJAZD',
		menuCount: n => `${n} dostępnych`,
		menuFootXr: 'prawy drążek - nawigacja   A - wybierz   B - zamknij',
		menuFootFlat: '[strzałki] - nawigacja   [Enter] - wybierz   [Esc] - zamknij',
		pts: 'pkt',

		thrOff: 'wył.',
		thrValue: (t, cnt) => `${t} (>=${cnt}/woksel)`,
		hudPoints: 'punkty:',
		hudVoxels: 'woksele:',
		hudMax: 'max:',
		hudMaxUnit: '/woksel',
		hudThreshold: 'próg (+/-):',
		hudSteps: 'kroki (Q/A):',
		hudSamples: 'prób/woksel',
		hudBuffer: 'bufor:',
		hudColour: 'M: tryb koloru',
		// The in-scene panel repeats a few of these without the key hints; there
		// is no keyboard in the headset to hint at.
		panelBuffer: 'bufor',
		panelSteps: 'kroki',
		panelThreshold: t => `próg ${t}`,
		panelLoading: dvc => `ładowanie dvc ${dvc}...`,
		statusLoading: dvc => `ładuję dvc=${dvc} …`,
		statusFailed: (dvc, m) => `nie udało się załadować dvc=${dvc}: ${m}`,

		xrButton: {
			'START AR': 'WŁĄCZ AR',
			'STOP AR': 'WYŁĄCZ AR',
			'AR NOT SUPPORTED': 'AR NIEDOSTĘPNE',
			'AR NOT ALLOWED': 'AR ZABLOKOWANE',
			'ENTER VR': 'WŁĄCZ VR',
			'EXIT VR': 'WYJDŹ Z VR',
			'VR NOT SUPPORTED': 'VR NIEDOSTĘPNE',
			'VR NOT ALLOWED': 'VR ZABLOKOWANE',
		},
	},

	en: {
		title: '3D acceleration density — XR',
		langOther: '\u{1F1F5}\u{1F1F1} Polski',   // regional indicators PL
		loadingVolume: 'loading volume…',
		helpFile: './VR_inputs.EN_en.txt',
		numLocale: 'en-GB',

		depthStages: ['source (nearest)', 'prepass: metres', 'prepass: occlusion'],
		depthArOnly: 'depth is only available in an AR session',
		depthUnavailable: 'depth-sensing unavailable on this device',
		depthNoPose: 'no viewer pose',
		depthNoFrame: 'no depth frame',
		depthError: m => `depth: ${m}`,
		depthStage: (i, name) => `depth ${i}: ${name}`,
		rulerNone: 'none',

		menuTitle: 'VEHICLE',
		menuCount: n => `${n} available`,
		menuFootXr: 'right stick - navigate   A - select   B - close',
		menuFootFlat: '[arrows] - navigate   [Enter] - select   [Esc] - close',
		pts: 'pts',

		thrOff: 'off',
		thrValue: (t, cnt) => `${t} (>=${cnt}/voxel)`,
		hudPoints: 'points:',
		hudVoxels: 'voxels:',
		hudMax: 'max:',
		hudMaxUnit: '/voxel',
		hudThreshold: 'threshold (+/-):',
		hudSteps: 'steps (Q/A):',
		hudSamples: 'samples/voxel',
		hudBuffer: 'buffer:',
		hudColour: 'M: colour mode',
		panelBuffer: 'buffer',
		panelSteps: 'steps',
		panelThreshold: t => `threshold ${t}`,
		panelLoading: dvc => `loading dvc ${dvc}...`,
		statusLoading: dvc => `loading dvc=${dvc} …`,
		statusFailed: (dvc, m) => `could not load dvc=${dvc}: ${m}`,

		xrButton: {},   // the vendored buttons already speak English
	},
};

export const T = STRINGS[LANG];

// three's ARButton/VRButton set their own label, from an async support check and
// again on every session event, so a one-off assignment would be overwritten.
// Watching the button and re-labelling afterwards leaves the vendored addons
// untouched. A replacement is never itself a key, so this settles in one pass.
export function localiseXrButton(el) {
	const map = T.xrButton;
	const apply = () => {
		const t = map[el.textContent];
		if (t) el.textContent = t;
	};
	new MutationObserver(apply).observe(el, {
		childList: true, characterData: true, subtree: true,
	});
	apply();
}
