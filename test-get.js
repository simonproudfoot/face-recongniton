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
const savedData = require("./savedFaceSearch.json");
//const tf = require('@tensorflow/tfjs');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
require('@tensorflow/tfjs-node');
const app = express()
const port = 3000
let allFaces = []
let refImage = ''
let loadedFaces = []
async function testResponse(loadedData) {
  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
  const url = 'https://lp-picture-library.greenwich-design-projects.co.uk/wp-content/uploads/2022/05/jonas-khan-headshot-239x300.jpg';
  const image = await canvas.loadImage(url);
  let content = JSON.parse(loadedData)
  const labeledFaceDescriptors = await Promise.all(content.map(className => {
    if (className) {
      // console.log(className)
      const descriptors = [];
      for (var i = 0; i < className.descriptors.length; i++) {
        descriptors.push(new Float32Array(className.descriptors[i]));
      }
      // let float = new Float32Array(descriptors)
       console.log(className.label)
      return new faceapi.LabeledFaceDescriptors(className.label, descriptors);
    }


  }))

  const faceMatcher = await new faceapi.FaceMatcher(labeledFaceDescriptors[0], 0.9) // this object is definatly the problerm
  //console.log(faceMatcher)







  // console.log('faceMatcher', faceMatcher) // mst likely incorrect formating or labeling
  const displaySize = { width: image.width, height: image.height }
  faceapi.matchDimensions(canvas, displaySize)
  const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
  // console.log('detections', detections)
  const resizedDetections = await faceapi.resizeResults(detections, displaySize)
  // console.log('resizedDetections', resizedDetections)
  const results = await resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor))
  console.log(results)

}

openFile()


async function openFile(data) {
  //var urlToCall = './savedFaceSearch.json';
  let rawdata = fs.readFileSync('./savedFaceSearch.json');

  testResponse(rawdata);


}

// app.listen(3000, () => {
//   console.log('app running on port: 3000');
//});