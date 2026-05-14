package com.aitrade.app;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(android.os.Bundle savedInstanceState) {
        registerPlugin(ZerodhaAutoLoginPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
