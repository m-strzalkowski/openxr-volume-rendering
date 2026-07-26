// Volume decoding and preprocessing, shared by main.js and volume-worker.js so
// the two can never drift apart. Nothing in here touches the DOM or WebGL.

export const HEADER = 64;

export function parseVolume(buf) {
	const dv = new DataView(buf);
	const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
	if (magic !== 'VOL1') throw new Error(`bad magic ${magic}`);
	const v = {
		version: dv.getUint32(4, true),
		nx: dv.getUint32(8, true),
		ny: dv.getUint32(12, true),
		nz: dv.getUint32(16, true),
		dtype: dv.getUint32(20, true),
		bound: dv.getFloat32(24, true),
		px: dv.getFloat32(28, true),
		logMax: dv.getFloat32(32, true),
		maxCount: dv.getUint32(36, true),
		points: Number(dv.getBigUint64(40, true)),
		rows: Number(dv.getBigUint64(48, true)),
		nonzero: dv.getUint32(56, true),
		dvc: dv.getUint32(60, true),
	};
	if (v.dtype !== 1) throw new Error('expected u8 payload');
	v.data = new Uint8Array(buf, HEADER, v.nx * v.ny * v.nz);
	return v;
}

// Bounding box of everything that survives each possible threshold.
//
// The payload is uint8, so there are only 256 of them. One pass records the
// extent of each distinct value; a suffix scan from 255 downwards then unions
// them into "extent of all voxels >= j" for every j. After that a threshold
// change is a lookup, so there is no asynchronous recompute to go stale and no
// need to treat raising and lowering the threshold differently.
export function buildThresholdBoxes(vol) {
	const { nx, ny, nz, data } = vol;
	const L = 256;
	const lo = new Int32Array(L * 3).fill(0x7fffffff);
	const hi = new Int32Array(L * 3).fill(-1);

	for (let z = 0; z < nz; z++) {
		for (let y = 0; y < ny; y++) {
			const row = (z * ny + y) * nx;
			for (let x = 0; x < nx; x++) {
				const v = data[row + x];
				if (v === 0) continue;
				const b = v * 3;
				if (x < lo[b]) lo[b] = x;
				if (x > hi[b]) hi[b] = x;
				if (y < lo[b + 1]) lo[b + 1] = y;
				if (y > hi[b + 1]) hi[b + 1] = y;
				if (z < lo[b + 2]) lo[b + 2] = z;
				if (z > hi[b + 2]) hi[b + 2] = z;
			}
		}
	}
	for (let j = L - 2; j >= 0; j--) {
		for (let k = 0; k < 3; k++) {
			const a = j * 3 + k, n = (j + 1) * 3 + k;
			if (lo[n] < lo[a]) lo[a] = lo[n];
			if (hi[n] > hi[a]) hi[a] = hi[n];
		}
	}

	// To texture coordinates, with a voxel of margin because trilinear taps
	// reach outside the voxel they land in.
	const dim = [nx, ny, nz];
	const boxes = new Array(L);
	for (let j = 0; j < L; j++) {
		if (hi[j * 3] < 0) { boxes[j] = null; continue; }   // nothing survives
		const min = [], max = [];
		for (let k = 0; k < 3; k++) {
			min.push(Math.max(0, (lo[j * 3 + k] - 1) / dim[k]));
			max.push(Math.min(1, (hi[j * 3 + k] + 2) / dim[k]));
		}
		boxes[j] = { min, max, diag: Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) };
	}
	return boxes;
}

// Fetch + decode + preprocess. Returns the raw buffer alongside the results so
// the caller can hand it to the GPU; `info` is the header without the sample
// data, which is what the HUD needs.
//
// The blob may arrive either way round. The live backend negotiates
// Content-Encoding, so fetch has already inflated it by the time we get here; a
// static export is a plain file that any dumb host serves verbatim, gzip and
// all. Neither end can be talked out of its habit -- Accept-Encoding is a
// forbidden request header and Content-Encoding is the server's to send -- so
// the payload identifies itself instead: VOL1 or 1f 8b. This also survives a
// host that re-compresses our file, since fetch strips exactly the transport
// layer and leaves ours.
export async function loadVolumeData(url) {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	let buf = await res.arrayBuffer();
	const magic = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
	if (magic[0] === 0x1f && magic[1] === 0x8b) {
		buf = await new Response(
			new Response(buf).body.pipeThrough(new DecompressionStream('gzip'))
		).arrayBuffer();
	}
	const vol = parseVolume(buf);
	const boxes = buildThresholdBoxes(vol);
	const info = { ...vol };
	delete info.data;
	return { info, boxes, buf };
}
