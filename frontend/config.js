// GradLab frontend configuration.
//
// Leave API_BASE empty ("") to run fully in-browser (optimiser playground works
// offline; classifier uses a lightweight in-browser model). Set it to your
// deployed Lambda Function URL or http://localhost:8000 to use the real Python
// engine for everything, including the AI tutor.
window.GRADLAB_CONFIG = {
  API_BASE: "",            // e.g. "https://abc123.lambda-url.ap-south-1.on.aws"
};
