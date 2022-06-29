const faceapi = require('face-api.js')
const express = require('express')
const path = require('node:path');
const fetch = require('cross-fetch');
const request = require('request');
const canvas = require("canvas");
const urllib = require('urllib')
const tf = require('@tensorflow/tfjs');
const https = require('https');
const http = require('http');
const fs = require('fs');
const { cos } = require('@tensorflow/tfjs');
require('@tensorflow/tfjs')
const base64 = require('node-base64-image')
const testData = require("./test.json");
//const tf = require('@tensorflow/tfjs');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
require('@tensorflow/tfjs-node');
const app = express()
const port = 3000

let allFaces = []
let refImage = ''
let loadedFaces = []


  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
  //const url = 'https://lp-picture-library.greenwich-design-projects.co.uk/wp-content/uploads/2022/04/georgia-and-tommy0755-final-rgb-1024x683.jpg';
  //const image = await canvas.loadImage(url);
  //console.log('searching image', image)

    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(
      labeledFaceDescriptors.filter(x => x != undefined),
      0.6
    );
    saveToFile(labeledFaceDescriptors)
  
}


async function loadLabeledImages() {
  console.log('loading faces')
  const data = await fetch('https://lp-picture-library.greenwich-design-projects.co.uk/wp-json/acf/v3/options/face-library').then((data) => data.json());
  const images = await data.acf['face-library']
  return Promise.all(
    images.map(async label => {
      const descriptions = []
      try {
        const img = await canvas.loadImage(label.image.sizes.medium)
        console.log('process image:', img)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
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
}

testResponse()


async function saveToFile(data) {
  var wstream = fs.createWriteStream('savedFaceSearch.json');
  wstream.write(JSON.stringify(data));
  wstream.end();
}








// app.listen(3000, () => {
//   console.log('app running on port: 3000');
//});