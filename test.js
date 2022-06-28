const express = require('express');
const faceapi = require('face-api.js');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas; 

faceapi.env.monkeyPatch({ fetch: fetch });

// SET STORAGE
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, '/uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  }
});
const upload = multer({ storage: storage });

const app = express();

app.post('/', async (req, res) => {
  console.log(req.file);

  try {
    const labeledFaceDescriptors = await loadLabeledImages();
    console.log('labeledFaceDescriptors: ', labeledFaceDescriptors);

    const faceMatcher = new faceapi.FaceMatcher(
      labeledFaceDescriptors,
      0.6
    );
    let image;
    image = await faceapi.bufferToImage(req.file);
    console.log(image);

    const displaySize = { width: image.width, height: image.height };
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(
      detections,
      displaySize
    );
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      res.send(result.toString());
    });
  } catch (error) {
    console.log(error);

    res.send(error.message);
  }
});

function loadLabeledImages() {
  const labels = [
    'Black Widow',
    'Captain America',
    'Captain Marvel',
    'Hawkeye',
    'Jim Rhodes',
    'Thor',
    'Tony Stark'
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img =  await canvas.loadImage(`http://localhost:3001/labeled_images/${label}/${i}.jpg`);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

app.listen(3000, () => {
  console.log('app running on port: 3000');
});