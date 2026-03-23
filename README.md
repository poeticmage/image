# Fundus Image Processing API

# Overview https://funduseyefrontend12.onrender.com/
# This API accepts a fundus image, validates it using one external service, and if accepted, sends it to another service for prediction. | Node.js Express Axios FormData CORS

# Client uploads an image
# Image is sent to API1 for validation
# If rejected, response is returned immediately
# If accepted, image is sent to API2
# Final prediction is returned

# API1: Checks whether the image is a Fundus image at all based on an ML application on DBSCAN clustering https://lioninthestreets-fundusimagegate.hf.space/check
# API2: Studies the image, extracts information, matches with the trained weights of a Light MaxViT based algorithm. https://lioninthestreets-maxvitgradcam.hf.space/predict
```
import express from "express";
import axios from "axios";
import FormData from "form-data";
import multer from "multer";
import serverless from "serverless-http"; 
import cors from "cors";

const app = express();
app.use(cors());
const upload = multer();

const API1 = "https://lioninthestreets-fundusimagegate.hf.space/check";
const API2 = "https://lioninthestreets-maxvitgradcam.hf.space/predict";

app.post("/image", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400);
    }
    const form = new FormData();
    form.append("file", req.file.buffer, req.file.originalname);
    const response1 = await axios.post(API1, form, {
      headers: form.getHeaders(),
    });
    console.log("API1:", response1.data);
    const status1 = response1.data.status?.toString().toLowerCase();
    if (status1 ==="reject") {
      return res.json({
        result: response1.data
      });
    }
    const form2 = new FormData();
    form2.append("file", req.file.buffer, req.file.originalname);
    const response2 = await axios.post(API2, form2, {
      headers: form2.getHeaders(),
    });
    console.log("API2:", response2.data);
    return res.json({
      result: response2.data,
    });
  } catch (e) {
    console.error(e);
    return res.status(500);
  }
});



const port = process.env.PORT || 3000;
app.listen(port, () => console.log("Running on", port));



// app.listen(3000, () => {
//   console.log("Server running on port 3000");
// });
```
