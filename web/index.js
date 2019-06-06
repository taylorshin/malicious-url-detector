const MAX_WORD_LENGTH = 45;
const TOTAL_NUM_WORDS = 515572.0;
const VOCAB_SIZE = 116085;
const LARGEST_VEC_LEN = 89;

async function loadModel() {
    console.log('Loading model...');
    const model = await tf.loadLayersModel('model.json');
    console.log('Model loaded.')
    return model;
}

function wordProb(dict, word) {
    return dict.hasOwnProperty(word) ? (dict[word] / TOTAL_NUM_WORDS) : 0;
}

function viterbiSegment(dict, text) {
    let probs = [1.0];
    let lasts = [0];
    for(let i = 1; i < text.length + 1; i++) {
        let prob_k_arr = [];
        let k_arr = [];
        for(let j = Math.max(0, i - MAX_WORD_LENGTH); j < i; j++) {
            prob_k_arr.push(probs[j] * wordProb(dict, text.slice(j, i)));
            k_arr.push(j);
        }
        // console.log('Max prob k:', Math.max(...prob_k_arr));
        let maxVal = Math.max(...prob_k_arr);
        let indexOfMaxVal = prob_k_arr.indexOf(maxVal);
        probs.push(Math.max(...prob_k_arr));
        lasts.push(k_arr[indexOfMaxVal]);
    }
    let words = [];
    let i = text.length;
    while (i > 0) {
        words.push(text.slice(lasts[i], i));
        i = lasts[i];
    }
    reversed = words.reverse();
    return [reversed, probs.slice(-1)[0]]
}

function tokenize(dict, url) {
    let tokens = url.split(new RegExp('[-/.&?=_]', 'g'));
    let tokensNew = [];
    tokens.forEach((token, index) => {
        let [wordSplit, prob] = viterbiSegment(dict, token);
        if (wordSplit.length < 4) {
            tokensNew = tokensNew.concat(wordSplit.slice(0, wordSplit.length));
        }
        else {
            tokensNew.push(token);
        }
    });
    return tokensNew;
}

function padArray(arr, len, fill) {
    return arr.concat(Array(len).fill(fill)).slice(0,len);
}
        
function predict(model, wordDict, tokenDict, url) {
    let tokens = tokenize(wordDict, url);
    console.log('tokens:', tokens);
    let int_seq = [];
    for (let token in tokens) {
        if (tokenDict.hasOwnProperty(token)) {
            int_seq.push(tokenDict[token]);
        }
        else {
            int_seq.push(0);
        }
    }
    // Pad array
    console.log('before:', int_seq);
    let pad_int_seq = padArray(int_seq, LARGEST_VEC_LEN, 0);
    console.log('after:',pad_int_seq);
    pad_int_seq = [pad_int_seq]
    const x = tf.tensor2d(pad_int_seq);
    console.log('x:', x);
    return model.predict(x);
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

const opts = {
    angle: 0, // The span of the gauge arc
    lineWidth: 0.6, // The line thickness
    radiusScale: 0.72, // Relative radius
    pointer: {
        length: 0.6, // // Relative to gauge radius
        strokeWidth: 0.033, // The thickness
        color: '#000000' // Fill color
    },
    limitMax: false,     // If false, max value increases automatically if value > maxValue
    limitMin: false,     // If true, the min value of the gauge will be fixed
    colorStart: '#6FADCF',   // Colors
    colorStop: '#8FC0DA',    // just experiment with them
    strokeColor: '#E0E0E0',  // to see which ones work best for you
    generateGradient: true,
    highDpiSupport: true,     // High resolution support
};

$(document).ready(function() {
    var target = document.getElementById('canvas'); // your canvas element
    var gauge = new Gauge(target).setOptions(opts); // create sexy gauge!
    gauge.maxValue = 3000; // set max gauge value
    gauge.setMinValue(0);  // Prefer setter over gauge.minValue = 0
    gauge.animationSpeed = 32; // set animation speed (32 is default value)
    gauge.set(850); // set actual value
});

loadModel().then((model) => {
    console.log('Model: ', model);
    let wordDict, tokenDict;
    $.when(
        $.getJSON('word_dict.json', (dict) => {
            wordDict = dict;
        }),
        $.getJSON('token_dict.json', (dict) => {
            tokenDict = dict;
        })
    ).then(() => {
        // let tokens = tokenize(dict, 'realinnovation.com/css/menu.js');
        // console.log('Tokens:', tokens);
        let pred = predict(model, wordDict, tokenDict, 'realinnovation.com/css/menu.js');
        console.log('Prediction:', pred.dataSync());
    });
});
