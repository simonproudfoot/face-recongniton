const faceapi = require('@vladmandic/face-api');
const express = require('express')
const path = require('node:path');
const fetch = require('cross-fetch');
//const request = require('request');
const canvas = require("canvas");
const urllib = require('urllib')
//const tf = require('@tensorflow/tfjs');
var cors = require('cors')
// const https = require('https');
// const http = require('http');
const fs = require('fs');
//const { cos } = require('@tensorflow/tfjs');
//require('@tensorflow/tfjs')
const base64 = require('node-base64-image')
const savedData = require("./savedFaceSearch.json");
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
//require('@tensorflow/tfjs-node');
const app = express()
let port = process.env.PORT || 3000
app.use(cors())
// FIND FACES
app.get('/find', async (req, res) => {
  faceapi.tf.engine().startScope();
  const url = req.query.imgUrl
  // if memory leak continiues. try moving these models out of function
  let faceRecognitionNet = await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
  let ssdMobilenetv1 = await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
  let faceLandmark68TinyNet = await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(path.join(__dirname, 'models'));
  let data = await savedData
  const image = await canvas.loadImage(url);
  let content = data
  const labeledFaceDescriptors = await Promise.all(content.map(className => {
    if (className) {
      const descriptors = [];
      for (var i = 0; i < className.descriptors.length; i++) {
        descriptors.push(new Float32Array(className.descriptors[i]));
      }
      return new faceapi.LabeledFaceDescriptors(className.label, descriptors);
    }
  }))
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors.filter(x => x != undefined)
  );
  const displaySize = { width: image.width, height: image.height }
  let faceDetectorOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
  faceapi.matchDimensions(canvas, displaySize)
  const detections = await faceapi.detectAllFaces(image, faceDetectorOptions).withFaceLandmarks(true).withFaceDescriptors()
  const resizedDetections = await faceapi.resizeResults(detections, displaySize)
  const results = await resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor))

  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(results));
  faceapi.tf.engine().endScope();
})


// UPDATE DATABASE
app.get('/update', async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(JSON.stringify({ result: 'done' }));
  faceapi.tf.engine().startScope();
  const url = req.query.imgUrl
  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
  // await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(path.join(__dirname, 'models'));
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors.filter(x => x != undefined),
    0.6
  );
  saveToFile(labeledFaceDescriptors)
  console.log('saved')
  faceapi.tf.engine().endScope();
})


async function loadLabeledImages() {

  const data = await fetch('https://lp-picture-library.greenwich-design-projects.co.uk/wp-json/acf/v3/options/face-library').then((data) => data.json());
  const images = await data.acf['face-library']
  return Promise.all(
    images.map(async label => {
      const descriptions = []
      try {
        const img = await canvas.loadImage(label.image.sizes.medium)
        console.log('process image:', img)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks(true).withFaceDescriptor()
        if (detections != undefined && detections.descriptor != undefined && label.name != undefined) {
          descriptions.push(detections.descriptor)
          return new faceapi.LabeledFaceDescriptors(label.name, descriptions);
        }
      } catch (error) {
        console.log('face error', error)
      }
    })
  )
}


async function saveToFile(data) {
  var wstream = fs.createWriteStream('savedFaceSearch.json');
  wstream.write(JSON.stringify(data));
  wstream.end();
}


app.listen(port, () => {
  console.log('app running on port: 3000');
});