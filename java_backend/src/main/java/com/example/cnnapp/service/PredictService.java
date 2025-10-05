package com.example.cnnapp.service;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;

@Service
public class PredictService
{
    public String predict(MultipartFile file) throws IOException
    {
        byte[] imageBytes = file.getBytes();

        return "Model received "+imageBytes.length+" bytes. (No ONNX model yet)";
    }
}