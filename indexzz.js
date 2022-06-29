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
//

app.get('/updateFaces', async (req, res) => {
  res.send('update local json file here')
})

app.get('/', async (req, res) => {
  const url = 'http://localhost:8888/lime-pictures-photo-library/wp/wp-content/uploads/2022/04/christian-buehner-DItYlc26zVI-unsplash-300x200.jpg';
  const dataUrl = 'http://localhost:8888/lime-pictures-photo-library/wp/wp-json/faces'
  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
  loadFacesFromDb(refImage, res)
})


// Global variable
var faceMatcher;

// Create Face Matcher
async function createFaceMatcher(data) {
  console.log('createFaceMatcher')
  const labeledFaceDescriptors = await Promise.all(data._labeledDescriptors.map(className => {
    const descriptors = [];
    for (var i = 0; i < className._descriptors.length; i++) {
      descriptors.push(className._descriptors[i]);
    }
    return new faceapi.LabeledFaceDescriptors(className._label, descriptors);
  }))
  return new faceapi.FaceMatcher(labeledFaceDescriptors);
}

// Load json to backend
async function loadFacesFromDb(blob, res) {
  console.log('createFaceMatcher')
  fs.readFile('./test.json', async function (err, data) {
    if (err) {
      console.log(err);
    }
    var content = JSON.parse(data);
    res.send(content)
    for (var x = 0; x < content['_labeledDescriptors'].length; x++) {
      for (var y = 0; y < content['_labeledDescriptors'][x]['_descriptors'].length; y++) {
        var results = Object.values(content['_labeledDescriptors'][x]['_descriptors'][y]);
        content._labeledDescriptors[x]._descriptors[y] = new Float32Array(results);
      }
    }
    faceMatcher = await createFaceMatcher(content);
    console.log(faceMatcher)
  });
}

// async function loadFacesFromDb(blob, res) {
//   const url = 'https://lp-picture-library.greenwich-design-projects.co.uk/wp-content/uploads/2022/04/chloe-brockett-crop-300x300.jpg';
//   const image = await canvas.loadImage(url);
//   const str = fs.readFileSync('./test.json')


//   // let obj = new Array(Object.values(JSON.parse(str.toString())))
//   // let arrayDescriptor = new Array(obj[0].length)
//   // let i = 0
//   // obj[0].forEach((entry) => {
//   //   arrayDescriptor[i++] = new Float32Array(Object.values(entry))
//   // });

//   // const faceMatcher = await new faceapi.FaceMatcher(arrayDescriptor) // this object is definatly the problerm
//   // console.log('faceMatcher', faceMatcher) // mst likely incorrect formating or labeling
//   // const displaySize = { width: image.width, height: image.height }
//   // faceapi.matchDimensions(canvas, displaySize)
//   // const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
//   // console.log('detections', detections)
//   // const resizedDetections = await faceapi.resizeResults(detections, displaySize)
//   // console.log('resizedDetections', resizedDetections)
//   // const results = await resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor))
//   // console.log(results)




// }

async function createFaceMatcher(data) {
  const labeledFaceDescriptors = await Promise.all(data.parent.map(className => {
    const descriptors = [];
    for (var i = 0; i < className._descriptors.length; i++) {
      descriptors.push(className._descriptors[i]);
    }
    return new faceapi.LabeledFaceDescriptors(className._label, descriptors);
  }))
  return new faceapi.FaceMatcher(labeledFaceDescriptors);
}


function loadImageData(urlToCall, callback) {
  console.log('loadImageData')
  urllib.request(urlToCall, { json: true, wd: 'nodejs' }, function (err, data, response) {
    var statusCode = response.statusCode;
    let finalData = data
    // call the function that needs the value
    callback(finalData);
    // we are done
    return;
  });
}

// async function getRefImage(urlToCall,) {
//   console.log('getRefImage')
//   const url = urlToCall;
//   const options = {
//     string: true,
//     headers: {
//       "User-Agent": "my-app"
//     }
//   };
//   // writing to file named 'example.jpg'
//   const image = await base64.encode(url, options);
//   return image
// }





















// async function loadAllImages() {
//     // try {
//     //     await loadLabeledImages();
//     //     let f = await JSON.parse(appParams.faces);
//     //     let a = []; await f.map((item, index) => {
//     //         if (item._descriptors[1] != undefined) {
//     //             let arr = []; arr[0] = new Float32Array(Object.keys(item._descriptors[0]).length);
//     //             arr[1] = new Float32Array(Object.keys(item._descriptors[1]).length);
//     //             for (let j = 0; j < 2; j++) { for (let i = 0; i < Object.keys(item._descriptors[j]).length; i++) { arr[j][i] = item._descriptors[j][i]; } } a[index] = new faceapi.LabeledFaceDescriptors(item._label, arr);
//     //         }
//     //     });
//     //     return allFaces
//     // } catch (err) {
//     //     console.error('error', err);
//     // }
//     // finally {
//     //     console.log('ran loadAllImages')
//     // }

// }


// // old
// async function loadLabeledImages(images) {
//     // console.log('loading faces')
//     // // const data = await fetch(appParams.siteUrl + '/wp-json/acf/v3/options/face-library').then((data) => data.json());
//     // // const images = await data.acf['face-library']
//     // return Promise.all(
//     //     images.map(async label => {
//     //         const descriptions = []
//     //         // try {

//     //         //     const detections = await faceapi.detectAllFaces(img)
//     //         //     console.log(img)
//     //         //     if (detections != undefined && detections.descriptor != undefined) {

//     //         //         console.log(detections.descriptor)
//     //         //         descriptions.push(detections.descriptor)
//     //         //         loadedFaces.push(new faceapi.LabeledFaceDescriptors(label.name, descriptions))
//     //         //     }
//     //         // } catch (error) {
//     //         //     console.log('face error', error)
//     //         // }

//     //         try {
//     //             const img = await canvas.loadImage(label.image.sizes.medium)
//     //             const desc = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
//     //             if (desc != undefined && desc.descriptor != undefined) {
//     //                 loadedFaces.push(Array.prototype.slice.call(desc.descriptor));
//     //             }
//     //         } catch (e) {
//     //             console.log(e);
//     //         }




//     //     })
//     // )
// }


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})



