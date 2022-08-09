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
const http = require('http');
const fs = require('fs');

const base64 = require('node-base64-image')
const savedData = require("./savedFaceSearch.json");

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let processing = false
//require('@tensorflow/tfjs-node');
const app = express()
let port = process.env.PORT || 3000
app.use(cors())

const server = http.createServer(app);
//const { Server } = require("socket.io");
const io = require('socket.io')(server, {
  cors: {
    origin: '*',
  }
});

let hasErrors = false

server.listen(port, () => {
  console.log('listening on *:' + port);
});


io.on('connection', async (socket) => {
  console.log('connected')

  socket.on("connect_error", (err) => {
    console.log(`connect_error due to ${err.message}`);
  });


  socket.on("updateFaces", async (from) => {
    if (!processing) {
      processing = true
      console.log('INCOMING REQUEST FROM: ' + from.from)
      faceapi.tf.engine().startScope();
      await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
      await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
      await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(path.join(__dirname, 'models'));
      const labeledFaceDescriptors = await loadLabeledImages(from.from, socket)

      hasErrors = labeledFaceDescriptors.find(x => x.status == 'rejected')

      // console.log(labeledFaceDescriptors.filter(x => x != undefined && x.status != 'rejected').map(y => y.value))


      const faceMatcher = new faceapi.FaceMatcher(
        labeledFaceDescriptors.filter(x => x != undefined && x.status != 'rejected').map(y => y.value),
        0.6
      );


      if (hasErrors) {
        //  socket.emit("errorMessage", 'Done, but with errors. Some images failed to load. Please check for missing images');
      }

      hasErrors = false
      processing = false


      saveToFile(labeledFaceDescriptors)
      faceapi.tf.engine().endScope();


    } else {
      socket.emit("errorMessage", 'Process already running. Please wait');
    }
  })

});



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

async function loadLabeledImages(url, socket) {

  const data = await fetch(url + '/wp-json/acf/v3/options/face-library').then((data) => data.json());
  const images = await data.acf['face-library']
  let total = 0
  socket.emit("totalFaces", images.length - 1);
  return Promise.allSettled(
    images.map(async label => {
      const descriptions = []
      total++
      // try 

      const img = await canvas.loadImage(label.image.sizes.medium)
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks(true).withFaceDescriptor()
      if (total % 4 == 0) {
        console.log(total)
        
        socket.emit("countDown", total);
      }
      console.log(img)
      if (img && detections != undefined && detections.descriptor != undefined && label.name != undefined) {
        descriptions.push(detections.descriptor)
        return new faceapi.LabeledFaceDescriptors(label.name, descriptions);
      }

    })
  )
}


async function saveToFile(data) {
  var wstream = fs.createWriteStream('savedFaceSearch.json');
  wstream.write(JSON.stringify(data));
  wstream.end();
}






