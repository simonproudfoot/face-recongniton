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
const { cos, image } = require('@tensorflow/tfjs');
const { disconnect } = require('node:process');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
let processing = false
//require('@tensorflow/tfjs-node');
const app = express()
let port = process.env.PORT || 3000
app.use(cors())

let allFaceData = []
let countdown = 0

const server = http.createServer(app);
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

  socket.on("thanks", (from) => {
    console.log('received')
  })

  socket.on('disconnect', function (event) {
    console.log('disconnected')

      let data = [{error: 'disconnected'}]
      var wstream = fs.createWriteStream('errorLog.json');
      wstream.write(JSON.stringify(data));
      wstream.end();


  })

  socket.on("updateFaces", async (from) => {
    if (!processing) {
      processing = true
      console.log('INCOMING REQUEST FROM: ' + from.from)
      faceapi.tf.engine().startScope();
      await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
      await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
      await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(path.join(__dirname, 'models'));
      loadLabeledImages(from.from, socket)
    }
  })

});


async function ProcessFaceData(labeledFaceDescriptors, socket) {
  let filtered = []
  labeledFaceDescriptors.forEach(face => {
    if (face != undefined) {
      filtered.push(face)
    }
  });

  const faceMatcher = new faceapi.FaceMatcher(
    filtered,
    0.6
  );

  hasErrors = false
  processing = false
  console.log('all done!')
  socket.emit("complete", true);
  socket.disconnect();
  faceapi.tf.engine().endScope();
  setTimeout(() => {
    saveToFile(filtered)
  }, 8000);

}

app.get('/test', async (req, res) => {
  res.send(`I'm here!!!`)
})

// FIND FACES
app.get('/find', async (req, res) => {
  faceapi.tf.engine().startScope();
  const url = req.query.imgUrl
  console.log('finding:' + url)
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
  console.log('found:' + results)
  res.setHeader('Content-Type', 'application/json');
  res.send(JSON.stringify(results));
  faceapi.tf.engine().endScope();
})

async function loadLabeledImages(url, socket) {

  const data = await fetch(url + '/wp-json/acf/v3/options/face-library').then((data) => data.json());
  const images = await data.acf['face-library']
  let total = 0
  socket.emit("totalFaces", images.length - 1);
  countdown = images.length
  socket.on("received", async (i) => {

    if (countdown > 0) {
      countdown--

      let label = images[countdown]
      if (label.image.filesize > 0 && label.image.mime_type == 'image/jpeg') {
        canvas.loadImage(label.image.url).then(async (img) => {


          const detections = await faceapi.detectSingleFace(img).withFaceLandmarks(true).withFaceDescriptor()
          if (detections != undefined && detections.descriptor != undefined && label.name != undefined) {
            let descriptions = []
            console.log(img)
            descriptions.push(detections.descriptor)
            allFaceData.push(new faceapi.LabeledFaceDescriptors(label.name, descriptions));
          } else {
            socket.emit('warningMessage', `Can't process ` + label.image.filename)

          }
        }).catch((er) => {
          socket.emit('errorMessage', `Can't load ` + label.image.filename)
          console.log(er)
        }).then(() => {
          socket.emit('countDown', countdown)
        }).catch((er) => {
          //socket('errorMessage', `Can't load ` + label.filename)
          console.log(er)
        })
      }
    } else {

      ProcessFaceData(allFaceData, socket)
    }
  })


}
async function saveToFile(data) {
  var wstream = fs.createWriteStream('savedFaceSearch.json');
  wstream.write(JSON.stringify(data));
  wstream.end();
}









