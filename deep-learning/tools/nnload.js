// Load assets/nn.js in node (no browser) to prototype playground behaviour:  const NN = require("./nnload.js");
const fs = require("fs"), path = require("path");
global.window = { devicePixelRatio: 1 };
global.document = { createElement: () => ({ getContext: () => ({}) }), documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: () => "" });
eval(fs.readFileSync(path.join(__dirname, "..", "assets", "nn.js"), "utf8"));
module.exports = window.NN;
