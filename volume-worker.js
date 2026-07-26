// Keeps the fetch and the 512 000-voxel threshold-box pass off the main thread.
// A dropped frame is invisible on a monitor and is judder in a headset, and the
// whole point of swapping vehicles this way is that the old cloud keeps
// rendering smoothly until the new one is ready.

import { loadVolumeData } from './volume.js';

self.onmessage = async ({ data: { id, url } }) => {
	try {
		const { info, boxes, buf } = await loadVolumeData(url);
		// Transferred, not copied: the worker gives up the buffer entirely.
		self.postMessage({ id, ok: true, info, boxes, buf }, [buf]);
	} catch (e) {
		self.postMessage({ id, ok: false, error: String(e?.message || e) });
	}
};
