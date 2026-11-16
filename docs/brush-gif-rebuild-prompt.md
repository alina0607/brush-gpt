# Brush GPT GIF Rebuild Prompt

Rebuild the selected Brush GPT cat animations from their saved model tokens. Use the local whole-face model output and saved E1 eye-part tokens; do not resample, train, or edit the image.

## Rendering steps

1. Parse every complete, decodable whole-face stroke from the saved token sequence. For each stroke endpoint, render that cumulative token prefix through `WholeFaceData.layers` and the canonical `brushgpt.tokens.render` path. This preserves the model painter’s stage and layer ordering.
2. Render the full face from the complete whole-face token sequence. Keep it as the completed-face handoff frame, with a short pause before the eye part begins.
3. Decode the E1 eye tokens into cumulative eye-stroke prefixes. For each prefix, create the displayed image with the same `with_eyes` semantics used by the project: remove the whole-face eye stage, insert the current eye prefix at the canonical eye-stage position, and replay all later face layers over it. Do not append a flattened finished-eye image or replace the whole canvas with a separate final PNG.
4. Keep one animation state for each whole-face stroke and each eye stroke. Start at the first whole-face stroke and end at the canonical combined result.
5. Render at the project’s normal canvas size, then resize the finished frame to the requested display size with LANCZOS. Keep the rendered RGB values; do not retouch, recolour, paste, or apply lossy colour-reduction filters. GIF palette encoding is the only format conversion.

## Validation

- The whole-face handoff frame must match the canonical whole-face render, allowing only GIF palette quantization.
- The last eye frame must match `render(with_eyes(face_layers, all_eye_strokes))`, allowing only GIF palette quantization.
- Inspect the transition to confirm the eye stage changes its intended region while the face keeps the canonical stage/layer brush texture.
- Save the rebuild into a new run folder first. Replace displayed assets only after the GIF is complete and validated.

## Current selections

The first corrected gallery update replaces Seed 0 #32 and #44 and Seed 3 #1, #5, and #8. They come from the existing saved token batches; each cat uses its saved sample once.
