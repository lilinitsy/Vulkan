/*
 * Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */
package de.saschawillems.vulkanSample;

import android.annotation.TargetApi;
import android.app.AlertDialog;
import android.app.NativeActivity;
import android.content.DialogInterface;
import android.content.pm.ApplicationInfo;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;

import java.io.IOException;
import java.util.concurrent.Semaphore;

import java.nio.file.Path;
import java.nio.file.Paths;  
import java.nio.file.StandardOpenOption;
import java.nio.file.Files;



public class VulkanActivity extends NativeActivity {

    static {
        // Load native library
        System.loadLibrary("native-lib");
    }
    @TargetApi(Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        Path environment_log = Paths.get(Environment.getExternalStorageDirectory() + "/" + "gltf_client.log");
        /*
        String tmpstr = "tmp";
        byte tmpbyte[] = tmpstr.getBytes();


        // Make sure the directory exists
        try {
            Files.createDirectories(environment_log);
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            Files.write(environment_log, tmpbyte, StandardOpenOption.CREATE); //Make the header
        } catch (IOException e) {
            e.printStackTrace();
        }
        */
    }

    // Use a semaphore to create a modal dialog

    private final Semaphore semaphore = new Semaphore(0, true);

    public void showAlert(final String message)
    {
        final VulkanActivity activity = this;

        ApplicationInfo applicationInfo = activity.getApplicationInfo();
        final String applicationName = applicationInfo.nonLocalizedLabel.toString();

        this.runOnUiThread(new Runnable() {
           public void run() {
               AlertDialog.Builder builder = new AlertDialog.Builder(activity, android.R.style.Theme_Material_Dialog_Alert);
               builder.setTitle(applicationName);
               builder.setMessage(message);
               builder.setPositiveButton("Close", new DialogInterface.OnClickListener() {
                   public void onClick(DialogInterface dialog, int id) {
                       semaphore.release();
                   }
               });
               builder.setCancelable(false);
               AlertDialog dialog = builder.create();
               dialog.show();
           }
        });
        try {
            semaphore.acquire();
        }
        catch (InterruptedException e) { }
    }
}
