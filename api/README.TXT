	================================================================
	===    RENDERING VOLUMES (3D textures) USING RAYMARCHING     ===
	===      AND META QUEST 3 DEPTH SENSOR TO DRIVE OPACITY      ===
	===               USING OPEN XR IN BROWSER	             ===
	===   IMPLEMENTED IN A DISGUSTING MANNER LARGELY BY AN LLM   ===
	================================================================

Nonexistent clouds in Augmented Reality (passthrough) into which you can put your hands into (literally).

3d texture is loaded into OpenGL and rendered using a raymarching shader.
Shader includes heavily postprocessed data from Meta Quest's depth sensor to drive opacity of background (cameras image).
Three.js is used for OpenGl Code.

WebXR is used and thus, this web page can be rendered:
	-> On desktop browser
	-> On Meta Quest browser as flat page
	-> As immersive experience (augmented reality with passthrough) on Meta Quest 3
		(On other headsets it also might work, although probably without depth sensing)
	-> On mobile devices
	-> On some mobile browsers also as Augemnted Reality (uses phone camera passthrough and accelerometers)

Live deployment:
	https://m-strzalkowski.github.io/openxr-volume-rendering/

No warranty included, see LICENSE.
