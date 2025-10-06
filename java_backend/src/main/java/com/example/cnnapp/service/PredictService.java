package com.example.cnnapp.service;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import com.microsoft.onnxruntime.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@Service
public class PredictService
{
    private static OrtEnviroment env;
    private static OrtSession session;

    static
    {
        try{
            env = OrtEnviroment.getEnviroment();
            session = env.createSession("model/model.onnx", new OrtSession.SessionOptions());
            System.out.println("ONNX model loaded successfully");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to load ONNX model");
        }
    }


    public String predict(MultipartFile file) throws IOException
    {
        if(file.isEmpty()) throw new IllegalArgumentException("Uploaded file is empty");

        BufferedImage img = ImageIO.read(file.getInputStream());
        if(img==null) throw new IllegalArgumentException("Invalid image file format");

        float[] inputData = preprocessImage(img);

        long[] shape = new long[]{1,1,28,28};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        OrtSession.Result output = session.run(Collections.singletonMap("input", inputTensor));

        float[][] resultArray = (float[][]) output.get(0).getValue();
        int predictedClass = argMax(resultArray[0]);

        String label = mapClassToLabel(predictedClass);

        return "Predicted class: "+label;
    }

    private float[] preprocessImage(BufferedImage img)
    {
        int width = 28;
        int height = 28;
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        resized.getGraphics().drawImage(img, 0, 0, width, height, null);

        float[] data = new float[width*height];
        int idx=0;

        for(int y=0; y<height; y++)
        {
            for(int x=0; x < width; x++)
            {
                int pixel = resized.getRGB(x,y) & 0xFF;
                data[idx++] = pixel/255.0f;
            }
        }
        return data;
    }

}