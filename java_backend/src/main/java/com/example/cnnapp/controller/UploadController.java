package com.example.cnnapp.controller;

import com.example.cnnapp.service.PredictService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class UploadController   
{ 

    @Autowired
    private PredictService predictService;

    @GetMapping("/")
    public String index()
    {
        return "index"; // loads templates/index.html
    }

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam("file") MultipartFile file, Model model)
    {
        try{
            String prediction = predictService.predict(file);
            model.addAttribute("result", prediction);
        } catch (Exception e) {
            model.addAttribute("result", "Error: "+e.getMessage());
        }
        return "index";
    }

}
