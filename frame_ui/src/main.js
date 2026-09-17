/**
 * The entry point: hand the page to the frame.
 *
 * This used to be the whole app — a probe that made one Request and dumped
 * render frames into a `<pre>`, to answer "does the browser reach the kernel"
 * before anything was built on it. That question is answered, and the widget
 * now asks it for itself, visibly, which is a better test than a page that only
 * ever agreed with itself.
 */

import "./style.css";
import { startFrame } from "./frame.js";

startFrame(document.getElementById("root"));
