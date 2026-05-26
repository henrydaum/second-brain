// Suggested prompts shown as chips on the first tutorial step. Edit freely —
// the tutorial randomly picks three from this pool each time it's built.
// Each entry: { prompt: <sent to the chat>, label: <shown on the chip> }.
const PROMPT_SUGGESTIONS = [
  // Iconic single-skill showcases — skill-name labels
  { prompt: "Show me the Mandelbrot Explorer.", label: "Mandelbrot set" },
  { prompt: "Show me Newton Basins with a Fisheye lens and Posterize To Palette. Also, make it green.", label: "Newton Basins" },
  { prompt: "Show me a Lorenz Attractor zoomed out a bit, with the Synthwave palette.", label: "Lorenz Attractor" },
  { prompt: "Run Conway's Life with the Aurora palette and add Bloom Glow.", label: "Conway's Life" },
  { prompt: "Show me Isometric Terrain with the Botanical palette.", label: "Isometric terrain" },
  { prompt: "Show me an Elementary Cellular Automaton with the Obsidian palette.", label: "Cellular automaton" },
  { prompt: "Show me an Attractor Cloud using the Clifford attractor with the Aurora palette.", label: "Attractor cloud" },

  // Background + object combos
  { prompt: "Add a Color Field background with a 3D Menger Sponge on top, and increase the depth to 3.", label: "3D Menger Sponge" },
  { prompt: "Show me Wave Sea with a 3D Crystal Cluster on top, in the Deep Sea palette.", label: "Crystal cluster on water" },
  { prompt: "Add a Color Field background, then add Typography saying 'HELLO' over it in the Synthwave palette.", label: "Typography poster" },
  { prompt: "Add a Color Field background and an L-System Crest on top, then a Kaleidoscope filter set to 8-fold symmetry.", label: "Eight-fold symmetry" },

  // Filter-heavy showcases — evocative labels
  { prompt: "Load the Julia Explorer and give it the 'Ink & Paper' palette. Then invert the colors with the Invert filter. Finally, add the Glitch Slice and Kaleidoscope filters.", label: "Oriental rug pattern" },
  { prompt: "Add the Wave Sea background, then change the palette to 'Synthwave.' After that, add the Pixel Sort and Dot Screen filters.", label: "80s synthwave" },
  { prompt: "Show me the Julia Explorer at the 'Dendrite' location, then add the Fisheye lens and Chromatic Aberration filters.", label: "Mythical fractal orb" },
  { prompt: "Show me the Terdragon variant of the Dragon Curve with the Anaglyph 3D filter.", label: "Trippy dragon curve" },
  { prompt: "Show me the L-System Grove as a Fern with a touch of Film Grain.", label: "Stylish artificial plant" },
  { prompt: "Show me the Burning Ship Explorer at the Antenna location with Scanlines and Chromatic Aberration.", label: "Retro arcade fractal" },
  { prompt: "Show me the Mandelbrot Explorer at Seahorse Valley, then Pixelate it heavily for an 8-bit look.", label: "8-bit Mandelbrot" },
  { prompt: "Show me Isometric Terrain with the Halftone filter and the Bauhaus palette.", label: "Newsprint terrain" },
  { prompt: "Show me the Julia Explorer at the Spiral location, then push the Levels Curve filter for high contrast.", label: "Moody Julia spiral" },
  { prompt: "Add Flow Streamlines with the Frost palette, then a 4-way Mirror filter.", label: "Symmetric flow field" },
  { prompt: "Show me Wave Sea with Posterize To Palette using the Coral Reef palette.", label: "Posterized sea" },
  { prompt: "Show me Isometric Terrain with the Glitch Slice filter and the Violet Storm palette.", label: "Glitched terrain" },
  { prompt: "Show me a Koch Snowflake with the Halftone filter and the Frost palette.", label: "Halftone snowflake" },
  { prompt: "Show me the Mandelbrot Explorer at Elephant Valley, then apply Watercolor for a painted look.", label: "Watercolor Mandelbrot" },
  { prompt: "Show me a Gray-Scott reaction-diffusion in the Maze regime, then apply Halftone.", label: "Reaction-diffusion maze" },
  { prompt: "Show me the Burning Ship Explorer at the Mast Spire with the Magma palette and a Vignette.", label: "Burning Ship fractal" },

  // Meta — 1:1 short prompts so the agent improvises
  { prompt: "Surprise me", label: "Surprise me" },
  { prompt: "Give me a suggestion", label: "Give me a suggestion" },
  { prompt: "Make me something weird and beautiful", label: "Something weird" },
  { prompt: "Interview me, then makes something based on my answers", label: "Interview me"}
];
