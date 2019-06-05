async function loadModel() {
    console.log('Loading model...');
    const model = await tf.loadLayersModel('model.json');
    console.log('Model loaded.')
    return model;
}
        
function predict(url) {
    // TODO: Add prediction logic to transform URL for model input
    // return model.predict(url);
    return 0.4;
}

function showPrediction(url) {
    const prediction = predict(url);

    const box = document.getElementById('prediction-box');
    const resultText = document.getElementById('prediction-result');
    if (prediction < 0.5) {
        resultText.innerText = 'safe';
        resultText.style.color = '#00bb00';
        box.style.borderColor = '#00bb00';
    }
    else {
        resultText.innerText = 'dangerous!'
        resultText.style.color = 'red';
        box.style.borderColor = 'red';
    }
    box.style.visibility = "visible";
}

loadModel().then((model) => {
    console.log('Model: ', model);
});
