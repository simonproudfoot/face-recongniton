const faceapi = require('face-api.js')
const express = require('express')
const path = require('node:path');
const fetch = require('cross-fetch');
const request = require('request');
const canvas = require("canvas");
const urllib = require('urllib')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs')
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
app.get('/', async (req, res) => {


    const url = 'https://lp-picture-library.greenwich-design-projects.co.uk/wp-content/uploads/2022/04/chloe-brockett-crop-300x300.jpg';
    const dataUrl = 'http://localhost:8888/lime-pictures-photo-library/wp/wp-json/faces'
    await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
    getRefImage(url, (refImage) => {
        loadImageData(dataUrl, (faceData) => {
            loadFacesFromDb(faceData).then(getFaces(refImage, res)).then((results) => {
                res.send('found:', results)
            })

            //   loadFacesFromDb(faceData)
            //   res.send(faceData)
            // loadLabeledImages(loaded).then((go) => {
            //     console.log(loadedFaces)
            // })
        })
    })
})


async function getFaces(img, res) {

   res.send('dne')


}

async function loadFacesFromDb(data) {
    //   await loadLabeledImages();
    let f = await JSON.parse(JSON.parse(data))
    let a = []; await f.map((item, index) => {
        if (item._descriptors[1] != undefined) {
            let arr = []; arr[0] = new Float32Array(Object.keys(item._descriptors[0]).length);
            arr[1] = new Float32Array(Object.keys(item._descriptors[1]).length);
            for (let j = 0; j < 2; j++) { for (let i = 0; i < Object.keys(item._descriptors[j]).length; i++) { arr[j][i] = item._descriptors[j][i]; } } a[index] = new faceapi.LabeledFaceDescriptors(item._label, arr);
        }
    });

    return true
}

function loadImageData(urlToCall, callback) {
    urllib.request(urlToCall, { json: true, wd: 'nodejs' }, function (err, data, response) {
        var statusCode = response.statusCode;
        let finalData = data
        // call the function that needs the value
        callback(finalData);
        // we are done
        return;
    });
}


function getRefImage(urlToCall, callback) {
    urllib.request(urlToCall, { wd: 'nodejs' }, function (err, data, response) {
        var statusCode = response.statusCode;
        let finalData = data
        // call the function that needs the value
        callback(finalData);
        // we are done
        return;
    });
}























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



