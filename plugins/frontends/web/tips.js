// Random tips shown in the empty chat. Edit freely — kept in their own file
// so adding/removing tips doesn't touch app logic.
//
// TIPS shows to everyone (signed in or not). TIPS_SIGNED_IN merges into the
// pool only once we know the visitor has an account, since these tips talk
// about account-only features (deleting shared canvases, the live
// code-writing toggle in account settings, credit purchasing, etc.).
const TIPS = [
  "Second Brain writes and executes code to make images — it doesn't use a diffusion image model.",
  "You can have up to six layers in a single canvas, each with up to four settings.",
  "Tap a canvas in the shared gallery or saved archive to remix it. This preserves the original layers and settings.",
  "You can completely avoid talking to an AI by using the manual controls.",
  "Conversations are ephemeral — your messages are cleared from the database whenever you start a new chat, or after 24 hours, whichever comes first.",
  "You can use ctrl/cmd + Z to undo changes, and ctrl/cmd + shift + Z to redo them.",
  "You can download at 2× resolution, or ½.",
  "Pressing 'Randomize' will generate a new image with a random seed, whereas 'Regenerate' will keep the same seed.",
  "If you hit 'Search,' you'll get technique results based on semantic similarity to your query — not just keywords.",
  "For any questions or feedback, please contact secondbrainservice@gmail.com",
  "Second Brain Art is built on top of Second Brain, a programmable open-source agentic framework made by Henry Daum, available at github.com/henrydaum/second-brain",
  "Second Brain has guardrails, and will refuse to generate images that aren't aligned with them."
];

const TIPS_SIGNED_IN = [
  "You can delete images you have shared from the gallery, and unsave something from your saved archive.",
  "The live code writing feature is available in the account settings. The agent tends to make more mistakes when using it, but the results are highly customizable.",
  "Cached renders are free; account credits are only spent when the app has to do new work.",
  "When using the live code writing feature, any techniques created will be available for any user to use, so long as they have the 'Include community techniques' option enabled. You can ask Second Brain to update and delete any techniques you've created.",
  "Your account page shows the current free-credit limits and top-up price — no subscriptions."
];
