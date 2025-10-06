package com.example.cnnapp.util;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnviroment;
import ai.onnxruntime.OrtException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;

public class ImagePreprocessor
{
    private static final int IMAGE_SIZE = 224;
    private static final int CHANNELS = 3;

    public static OnnxTensor preprocessImage(String imagePath, OrtEnviroment env) throws IOException, OrtException
    {
        BufferedImage img = ImageIO.read(new File(imagePath));
        if(img == null)
        {
            throw new IOException("Failed to read image "+imagePath);
        }

        BufferedImage rgbImage = new BufferedImage( IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = rgbImage.createGraphics();
        g.drawImage(resizeImage(img, IMAGE_SIZE, IMAGE_SIZE), 0, 0, null);
        g.dispose();

        float[] imgData = new float[CHANNELS * IMAGE_SIZE * IMAGE_SIZE];
        int idx = 0;

        for(int y=0; y<IMAGE_SIZE; y++)
        {
            for(int x=0; x<IMAGE_SIZE; x++)
            {
                int rgb = rgbImage.getRGB(x,y);

                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float gC = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;

                imgData[idx++] = r;
                imgData[idx++] = gC;
                imgData[idx++] = b;
            }
        }

        long[] shape = new long[]{1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE};
        FloatBuffer buffer = FloatBuffer.wrap(imgData);
        return OnnxTensor.createTensor(env, buffer, shape);
    }


    private static Image resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight)
    {
        return originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
    }
}