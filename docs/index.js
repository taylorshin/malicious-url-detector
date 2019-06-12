const MAX_WORD_LENGTH = 20;       // from Viterbi stuff: MAX_WORD_LENGTH
const TOTAL_NUM_WORDS = 222148;   // from Viterbi stuff: TOTAL_NUM_WORDS
const LARGEST_VEC_LEN = 32;       // from constants: max_num_tokens
const MAX_FEATURES = 8192;        // from constants: max_features

var wordDict = null;              // from Viterbi stuff: dictionary
var tokenDict = null;             // from convert_tokens_to_ints: token_dict
var model = null;
var gauge = null;

async function loadModel() {
    console.log('Loading model...');
    model = await tf.loadLayersModel('model.json');
    console.log('Model loaded.')
    return model;
}

function wordProb(dict, word) {
    return dict.hasOwnProperty(word) ? (dict[word] / TOTAL_NUM_WORDS) : 0;
}

function viterbiSegment(dict, url) {
    let split_url = url.split(/\/|\-|\.|\&|\?|\=|\_/);
    let results = [];

    split_url.forEach(function(text) {
        let probs = [1.0];
        let lasts = [0];
        for(let i = 1; i < text.length + 1; i++) {
            let prob_k_arr = [];
            let k_arr = [];
            for(let j = Math.max(0, i - MAX_WORD_LENGTH); j < i; j++) {
                prob_k_arr.push(probs[j] * wordProb(dict, text.slice(j, i)));
                k_arr.push(j);
            }
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
        let reversed = words.reverse();

        let singles = 0;
        reversed.forEach(function(word) {
            if (word.length === 1) {
                singles++;
            }
        });

        let result = '';
        if (singles === words.length) {
            result = reversed.join('');
        }
        else if (singles > 3) {
            result = reversed.join('');
        }
        else {
            result = reversed.join('.');
        }
        results.push(result);
    });

    let joined = results.join('.');
    return joined.split('.');
}

function convert_tokens_to_ints(tokens) {
    // Assumes tokens is array of tokens for single URL
    let int_seq = [];
    tokens.forEach(function(token) {
        if (!(tokenDict.hasOwnProperty(token))) {
            tokenDict[token] = Object.keys(tokenDict).length;
        }
        int_seq.push(tokenDict[token] % MAX_FEATURES);
    });
    let x = padArray(int_seq, LARGEST_VEC_LEN, 0);
    return x;
}

function padArray(arr, len, fill) {
    let arr_len = arr.length;
    if (arr.length > len) {
        return arr.slice(0, len);
    }
    else {
        return (Array(len).fill(fill).slice(0,len-arr_len)).concat(arr);
    }
}

function predict(model, wordDict, tokenDict, url) {
    let tokens = viterbiSegment(wordDict, url);
    let int_seq = convert_tokens_to_ints(tokens);
    const x = tf.tensor2d([int_seq]);
    let result = model.predict(x);
    return result.dataSync();
}

function showPrediction() {
    let url = document.getElementById('url').value;
    const prediction = predict(model, wordDict, tokenDict, url);
    console.log('Prediction received! Probability: ' + prediction);

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
    gauge.set(3000 * prediction); // set actual value
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
    percentColors: [[0.0, "#00bb00" ], [0.50, "#f9c802"], [1.0, "#ff0000"]]
};

$(document).ready(function() {
    var target = document.getElementById('canvas'); // your canvas element
    gauge = new Gauge(target).setOptions(opts); // create gauge!
    gauge.maxValue = 3000; // set max gauge value
    gauge.setMinValue(0);  // Prefer setter over gauge.minValue = 0
    gauge.animationSpeed = 32; // set animation speed (32 is default value)
    gauge.set(0); // set actual value

    var form = document.getElementById('form');
    function handleForm(event) {
        event.preventDefault();
    }
    form.addEventListener('submit', handleForm);
});

loadModel().then((model) => {
    console.log('Model: ', model);
    $.when(
        $.getJSON('word_dict.json', (dict) => {
            wordDict = dict;
        }),
        $.getJSON('token_dict.json', (dict) => {
            tokenDict = dict;
        })
    ).then(() => {
        // Start allowing predictions from textbox
        document.getElementById('submit_button').disabled = false;
        document.getElementById('loading').style.visibility = 'hidden';
    });
});
