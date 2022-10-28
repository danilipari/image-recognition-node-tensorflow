import express from "express";
import cors from "cors";
import helmet from "helmet";
import os from "os";
import morgan from "morgan";

import * as tf from "@tensorflow/tfjs-node";
import coco_ssd from "@tensorflow-models/coco-ssd";
import busboy from "busboy";
/* require('dotenv').config(); */
import { config } from "dotenv";
config();

/*
  https://betterprogramming.pub/node-js-implementation-of-image-recognition-using-tensorflow-and-express-js-b006f5609415
*/

  /**
   * @description Init Model
  */
  let model = undefined;
  (async () => {
    model = await coco_ssd.load({
      base: "mobilenet_v1",
    });
  })();

  /**
   * @description Init Express
  */
  const app = express();
  app.use(helmet());
  app.disable('x-powered-by');
  app.use(express.urlencoded({extended: true}));
  app.use(express.json());
  app.use(morgan('combined'));
  const PORT = process.env.PORT || 8081;
  app.use(express.json());

  app.use(cors());

  /* Homepage Backend */
  app.get('/', (req, res) =>
    res.send(`<h2">Image Recognition TensorFlow System</h2>`)
  );

  app.post("/predict", (req, res) => {
    if (!model) {
      res.status(500).send("Model is not loaded yet!");
      return;
    }

    /**
     * @description Create a Busboy instance
    */
    const bb = busboy({ headers: req.headers });
    bb.on("file", (fieldname, file, filename, encoding, mimetype) => {
      console.log(file, '---file---');
      const buffer = [];
      file.on("data", (data) => {
        buffer.push(data);
      });
      file.on("end", async () => {
        const image = tf.node.decodeImage(Buffer.concat(buffer));
        const predictions = await model.detect(image, 5, 0.001);
        console.log(predictions, '---predictions---');
        res.json(predictions);
      });
    });
    req.pipe(bb);
  });

  const port = process.env.PORT || 8081;
  app.listen(port, () => { console.log(`Server is running on port ${port}.`) });