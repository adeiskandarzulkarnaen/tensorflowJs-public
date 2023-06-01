const tf = require('@tensorflow/tfjs-node');

async function run() {
  /* training data */
  const trainingUrl = 'file://src/data/wdbc-train.csv';
  const trainingData = tf.data.csv(trainingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true
      }
    }
  });
  const convertedTrainingData = trainingData.map(({xs, ys}) => {
    return {
      xs: Object.values(xs),
      ys: Object.values([ys.diagnosis])
    }
  }).batch(10);
  
  
  /* testing data */
  const testingUrl = 'file://src/data/wdbc-test.csv';
  const testingData = tf.data.csv(testingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true
      }
    }
  });
  const convertedTestingData = testingData.map(({xs, ys}) => {
    return {
      xs: Object.values(xs),
      ys: Object.values([ys.diagnosis])
    }
  }).batch(10);
  
  
  /* modeling */
  const numOfFeatures = (await trainingData.columnNames()).length - 1
  
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 31, inputShape: [numOfFeatures], activation: "relu" }),
      tf.layers.dense({ units: 16, activation: "relu" }),
      tf.layers.dense({ units: 8, activation: "relu" }),
      tf.layers.dense({ units: 4, activation: "relu" }),
      tf.layers.dense({ units: 2, activation: "relu" }),
      tf.layers.dense({ units: 1, activation: "sigmoid" })
    ]
  });
  
  model.compile({
    loss: "binaryCrossentropy",
    optimizer: tf.train.rmsprop(0.07),
    metrics:['accuracy']
  });
  
  
  /* train */
  await model.fitDataset(convertedTrainingData, {
    epochs: 100,
    validationData: convertedTestingData,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch} Loss: ${logs.loss} Accuracy: ${logs.acc}`);
      }
    }
  });
  
  /* save model */
  // await model.save('file://src/data/my-model');
}

run();