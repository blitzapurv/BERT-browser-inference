/** */
/*global BigInt */
/*global BigInt64Array */

import { loadTokenizer } from './bert_tokenizer.ts';
import * as wasmFeatureDetect from 'wasm-feature-detect';

// tokenizer
const { AutoTokenizer } = require('@xenova/transformers');
// Setup onnxruntime 
const ort = require('onnxruntime-web');

// requires Cross-Origin-*-policy headers https://web.dev/coop-coep/
const options = {
  executionProviders: ['wasm'], 
  graphOptimizationLevel: 'all'
};

var downLoadingModel = true;
const model = "./saved_onnx/classifier.onnx"; // Update with the correct model path

const session = ort.InferenceSession.create(model, options);
session.then(t => { 
  downLoadingModel = false;
  // warmup the VM
  for (var i = 0; i < 10; i++) {
    console.log("Inference warmup " + i);
    lm_inference("this is a warmup inference");
  }
});

const tokenizer = loadTokenizer();

const DEFAULT_DISPLAY = [
  ["Classification", "Score"],
  ['irrelevant', 0],
  ['relevant', 0],
];

const LABELS = [
  'irrelevant',
  'relevant',
];

function isDownloading() {
  return downLoadingModel;
}

function sigmoid(t) {
  return 1/(1+Math.pow(Math.E, -t));
}

function create_model_input(encoded) {
  var input_ids = new Array(encoded.length + 2);
  var attention_mask = new Array(encoded.length + 2);
  var token_type_ids = new Array(encoded.length + 2);
  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  var i = 0;
  for (; i < encoded.length; i++) { 
    input_ids[i + 1] = BigInt(encoded[i]);
    attention_mask[i + 1] = BigInt(1);
    token_type_ids[i + 1] = BigInt(0);
  }
  input_ids[i + 1] = BigInt(102);
  attention_mask[i + 1] = BigInt(1);
  token_type_ids[i + 1] = BigInt(0);
  const sequence_length = input_ids.length;
  input_ids = new ort.Tensor('int64', BigInt64Array.from(input_ids), [1, sequence_length]);
  attention_mask = new ort.Tensor('int64', BigInt64Array.from(attention_mask), [1, sequence_length]);
  token_type_ids = new ort.Tensor('int64', BigInt64Array.from(token_type_ids), [1, sequence_length]);
  return {
    input_ids: input_ids,
    attention_mask: attention_mask,
    token_type_ids: token_type_ids
  }
}

async function lm_inference(text) {
  try { 
    const encoded_ids = await tokenizer.then(t => {
      return t.tokenize(text); 
    });
    console.log(encoded_ids)
    if (encoded_ids.length === 0) {
      return [0.0, DEFAULT_DISPLAY];
    }
    const start = performance.now();
    const model_input = create_model_input(encoded_ids);
    const output = await session.then(s => { return s.run(model_input, ['output_0']) });
    const duration = (performance.now() - start).toFixed(1);
    const probs = output['output_0'].data.map(sigmoid).map(t => Math.floor(t * 100));
    
    const result = [];
    for (var i = 0; i < LABELS.length; i++) {
      const t = [LABELS[i], probs[i]];
      result[i] = t;
    }
    
    const result_list = [];
    result_list[0] = ["Classification", "Score"];
    for (i = 0; i < result.length; i++) {
      result_list[i + 1] = result[i];
    }
    return [duration, result_list];    
  } catch (e) {
    console.log(e)
    return [0.0, DEFAULT_DISPLAY];
  }
}    

async function loadTokenizer_new() {
  // Load the tokenizer
  const tokenizer = await AutoTokenizer.from_pretrained(`./saved_onnx/tokenizer.json`);

  return tokenizer;
}

export let inference = lm_inference;
export let columnNames = DEFAULT_DISPLAY;
export let modelDownloadInProgress = isDownloading;
