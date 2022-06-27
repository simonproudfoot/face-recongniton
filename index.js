


const faceapi = require('face-api.js')
const express = require('express')
const path = require('node:path');
const fetch = require('cross-fetch');
const request = require('request');
const canvas = require("canvas");
const urllib = require('urllib')
const tf = require('@tensorflow/tfjs');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
require('@tensorflow/tfjs-node');
const app = express()
const port = 3000

let allFaces = []
let refImage = ''
let loadedFaces = []


app.get('/faces', async (req, res) => {
    const url = 'https://lp-picture-library.greenwich-design-projects.co.uk/wp-content/uploads/2022/04/chloe-brockett-crop-300x300.jpg';
    await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));


    //  accordingly for the other models:
    await faceapi.loadTinyFaceDetectorModel('/models')
    await faceapi.loadMtcnnModel('/models')
    await faceapi.loadFaceLandmarkModel('/models')
    await faceapi.loadFaceLandmarkTinyModel('/models')
    await faceapi.loadFaceRecognitionModel('/models')
    await faceapi.loadFaceExpressionModel('/models')




    getRefImage(url, function (refImage) {
        //res.set("Content-Type", "image/jpeg");
        //  res.send(refImage)
        loadAllImages().then((allFaces) => {
            loadLabeledImages(allFaces)
            res.send('done!')
        })


    })
})


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

async function loadAllImages() {

    try {
        const data = await fetch('https://lp-picture-library.greenwich-design-projects.co.uk/wp-json/acf/v3/options/face-library');
        if (data.status >= 400) {
            throw new Error("Bad response from server");
        }
        let final = await data.json()
        allFaces = final['acf']['face-library']

        return allFaces

        //  let faces = await loadLabeledImages(images)
        //    res.send('loaded images and reference')
    } catch (err) {
        console.error('error', err);
    }
    finally {
        console.log('ran loadAllImages')
    }

}


// old
async function loadLabeledImages(images) {
    console.log('loading faces')
    // const data = await fetch(appParams.siteUrl + '/wp-json/acf/v3/options/face-library').then((data) => data.json());
    // const images = await data.acf['face-library']
    return Promise.all(
        images.map(async label => {
            const descriptions = []
            try {
                const img = await canvas.loadImage(label.image.sizes.medium)
                
                const detections = await faceapi.detectAllFaces(img)
                if (detections != undefined && detections.descriptor != undefined) {
                    descriptions.push(detections.descriptor)
                    loadedFaces.push(new faceapi.LabeledFaceDescriptors(label.name, descriptions))
                }
            } catch (error) {
                console.log('face error', error)
            }
        })
    )
}


app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})
