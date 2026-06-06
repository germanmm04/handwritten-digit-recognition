// Configuración del modelo ONNX
const MODEL_URL = "./models/mnist_cnn_32x32.onnx";

let session = null;
let isModelReady = false;

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

const predictionDigitEl = document.getElementById("predictionDigit");
const modelStatusEl = document.getElementById("modelStatus");

let drawing = false;
let strokeWidth = 24;

// Inicializar lienzo: fondo negro, trazo blanco (igual que MNIST)
function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "white";
  ctx.lineWidth = strokeWidth;
}

initCanvas();

function setStatus(text, type) {
  modelStatusEl.textContent = text;
}

async function loadModel() {
  try {
    setStatus("Inicializando sesión ONNX…", "loading");
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
    isModelReady = true;
    setStatus("Modelo cargado correctamente.", "ok");
  } catch (err) {
    console.error(err);
    setStatus("No se pudo cargar el modelo ONNX.", "error");
  }
}

loadModel();

// Eventos de dibujo (ratón y táctil)
function getCanvasPos(event) {
  const rect = canvas.getBoundingClientRect();
  if (event.touches && event.touches.length > 0) {
    return {
      x: ((event.touches[0].clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.touches[0].clientY - rect.top) * canvas.height) / rect.height,
    };
  }
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function startDrawing(e) {
  drawing = true;
  const pos = getCanvasPos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
  if (!drawing) return;
  const pos = getCanvasPos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function stopDrawing() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startDrawing(e);
});
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  draw(e);
});
canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  stopDrawing();
});

clearBtn.addEventListener("click", () => {
  initCanvas();
  predictionDigitEl.textContent = "?";
});

// Preprocesado: capturar, escalar a 32x32, gris, normalizar [0,1] y a tensor [1,32,32,1]
function preprocessImage() {
  const targetSize = 32;

  // Canvas auxiliar en memoria para redimensionar a 32x32
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = targetSize;
  tempCanvas.height = targetSize;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.fillStyle = "black";
  tempCtx.fillRect(0, 0, targetSize, targetSize);
  tempCtx.drawImage(canvas, 0, 0, targetSize, targetSize);

  const imgData = tempCtx.getImageData(0, 0, targetSize, targetSize);
  const pixels = imgData.data;

  const gray = new Float32Array(targetSize * targetSize);

  for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    // Escala de grises estándar
    let v = 0.299 * r + 0.587 * g + 0.114 * b; // [0,255]

    // Como MNIST tiene dígitos blancos sobre fondo negro, y nuestro canvas
    // ya dibuja en blanco sobre negro, NO invertimos aquí.
    // Si tuvieras fondo blanco y trazo negro, usarías: v = 255 - v;

    v = v / 255.0; // [0,1]
    gray[j] = v;
  }

  // Crear tensor [1,32,32,1] en orden NHWC
  const input = new Float32Array(targetSize * targetSize * 1);
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const idx = y * targetSize + x;
      const dstIdx = idx; // canal único (grayscale)
      input[dstIdx] = gray[idx];
    }
  }

  const tensor = new ort.Tensor("float32", input, [1, targetSize, targetSize, 1]);
  return tensor;
}

async function runInference() {
  if (!isModelReady || !session) {
    alert("El modelo todavía no está listo. Espera un momento.");
    return;
  }

  try {
    predictBtn.disabled = true;
    setStatus("Calculando predicción…", "loading");

    const inputTensor = preprocessImage();

    const feeds = { input: inputTensor };
    const outputMap = await session.run(feeds);

    const outputName = Object.keys(outputMap)[0];
    const logits = outputMap[outputName].data;

    // Softmax ya está aplicado en el modelo (salida con 'softmax'),
    // pero por si acaso normalizamos.
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    const exps = logits.map((v) => Math.exp(v - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sumExps);

    let bestDigit = 0;
    let bestProb = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestDigit = i;
      }
    }

    predictionDigitEl.textContent = bestDigit.toString();
    setStatus("Predicción lista.", "ok");
  } catch (err) {
    console.error(err);
    setStatus("Error durante la inferencia.", "error");
  } finally {
    predictBtn.disabled = false;
  }
}

predictBtn.addEventListener("click", () => {
  runInference();
});
