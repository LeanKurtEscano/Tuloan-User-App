import React, { useRef, useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, Alert, ActivityIndicator, Animated } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';


export default function LiveFaceVerification() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [idUploaded, setIdUploaded] = useState<boolean>(false);
  const [matchStatus, setMatchStatus] = useState<string>("Position your face");
  const [uploadingId, setUploadingId] = useState<boolean>(false);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [isMatched, setIsMatched] = useState<boolean>(false);
  const [confidence, setConfidence] = useState<number>(0);
  
  // Animation for scanning line
  const scanAnimation = useRef(new Animated.Value(0)).current;

  // Replace with your PC's LAN IP or emulator IP
  const serverUrl = "http://192.168.1.12:8000/api/facial/v1";

  // Request camera permission on mount
  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, []);

  // Scanning line animation
  useEffect(() => {
    if (isComparing) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scanAnimation, {
            toValue: 1,
            duration: 1500,
            useNativeDriver: true,
          }),
          Animated.timing(scanAnimation, {
            toValue: 0,
            duration: 0,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [isComparing]);

  // Automatic live verification interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (idUploaded && cameraRef.current && !isComparing) {
      interval = setInterval(() => {
        captureAndCompare();
      }, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [idUploaded, isComparing]);

  // Upload ID image
  const uploadId = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        quality: 0.8,
        aspect: [3, 4],
      });

      if (result.canceled || !result.assets?.length) return;

      setUploadingId(true);
      const formData = new FormData();
      
      const uri = result.assets[0].uri;
      const filename = uri.split('/').pop() || 'id.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('file', {
        uri: uri,
        name: filename,
        type: type,
      } as any);

      const response = await axios.post(`${serverUrl}/upload-id`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;
      
      if (data.status === 'success') {
        setIdUploaded(true);
        setMatchStatus("Position your face in the frame");
        Alert.alert("Success", "ID uploaded! Face verification will start automatically.");
      } else {
        Alert.alert("Error", data.message || "Failed to upload ID. Please use a clear photo with your face visible.");
      }
    } catch (err) {
      console.error('Upload ID Error:', err);
      Alert.alert("Error", "Could not upload ID. Check server connection.");
    } finally {
      setUploadingId(false);
    }
  };

  // Capture live frame and compare
  const captureAndCompare = async () => {
    if (!cameraRef.current || isComparing) return;

    setIsComparing(true);

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        skipProcessing: false,
      });

      if (!photo?.uri) {
        setIsComparing(false);
        return;
      }

      const formData = new FormData();
      const filename = photo.uri.split('/').pop() || 'live.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('file', {
        uri: photo.uri,
        name: filename,
        type: type,
      } as any);

      const response = await axios.post(`${serverUrl}/compare`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;
      
      if (data.no_id) {
        setMatchStatus("⚠️ Please upload ID first");
        setIsMatched(false);
      } else if (data.no_face_detected) {
        setMatchStatus("👤 No face detected");
        setIsMatched(false);
        setConfidence(0);
      } else if (data.match) {
        setIsMatched(true);
        const confidencePercent = Math.max(0, Math.min(100, (1 - data.distance / data.threshold) * 100));
        setConfidence(confidencePercent);
        setMatchStatus(`✅ Face Verified!`);
      } else {
        setIsMatched(false);
        setConfidence(0);
        setMatchStatus(`❌ Face does not match`);
      }
    } catch (err) {
      console.error('Compare Error:', err);
      setIsMatched(false);
      setMatchStatus("⚠️ Connection error");
    } finally {
      setIsComparing(false);
    }
  };

  if (!permission) {
    return (
      <View className="flex-1 justify-center items-center bg-gray-900">
        <ActivityIndicator size="large" color="#3B82F6" />
        <Text className="text-white text-lg mt-4">Loading camera...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View className="flex-1 justify-center items-center bg-gradient-to-b from-blue-900 to-gray-900 p-6">
        <View className="bg-white/10 p-8 rounded-3xl items-center">
          <Text className="text-white text-2xl font-bold mb-4 text-center">
            Camera Access Required
          </Text>
          <Text className="text-white/80 text-base mb-8 text-center">
            We need camera permission for face verification
          </Text>
          <TouchableOpacity
            onPress={requestPermission}
            className="bg-blue-600 px-10 py-4 rounded-full active:bg-blue-700"
          >
            <Text className="text-white font-bold text-lg">Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const scanTranslateY = scanAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 320], // Height of the circular frame (80 * 4 = 320)
  });

  return (
    <View className="flex-1 bg-black">
      {!idUploaded ? (
        <View className="flex-1 justify-center items-center bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
          <View className="bg-white rounded-3xl p-8 shadow-2xl items-center w-full max-w-md">
            <View className="bg-blue-100 p-6 rounded-full mb-6">
              <Text className="text-5xl">🆔</Text>
            </View>
            <Text className="text-3xl font-bold text-gray-900 mb-3 text-center">
              Face Verification
            </Text>
            <Text className="text-base text-gray-600 mb-8 text-center">
              Upload your ID photo to start the verification process
            </Text>
            <TouchableOpacity
              onPress={uploadId}
              disabled={uploadingId}
              className={`w-full px-8 py-4 rounded-2xl ${
                uploadingId ? 'bg-gray-400' : 'bg-blue-600 active:bg-blue-700'
              }`}
            >
              <Text className="text-white font-bold text-lg text-center">
                {uploadingId ? "Uploading..." : "📸 Upload ID Photo"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View className="flex-1 bg-white">
          {/* Header */}
          <View className="pt-12 pb-4 px-6 bg-blue-600">
            <Text className="text-white text-xl font-bold text-center">
              Login Authentication
            </Text>
          </View>

          {/* Main Content Area - White Background */}
          <View className="flex-1 bg-white px-6 pt-8">
            {/* Instructions at top */}
            <View className="mb-4">
              <Text className="text-gray-900 text-xl font-bold text-center mb-2">
                Hold the phone still. Rotate your Head. Go slow
              </Text>
              <Text className="text-gray-700 text-base text-center mb-1">
                Position your face in the circle
              </Text>
              <Text className="text-gray-600 text-sm text-center">
                Face detected. Please look into the circle and stay still.
              </Text>
            </View>

            {/* Centered Camera Preview Circle */}
            <View className="flex-1 justify-center items-center">
              <View className="relative" style={{ width: 320, height: 320 }}>
                {/* Circular Camera View Container */}
                <View 
                  style={{
                    width: 320,
                    height: 320,
                    borderRadius: 160,
                    overflow: 'hidden',
                    shadowColor: '#000',
                    shadowOffset: { width: 0, height: 10 },
                    shadowOpacity: 0.15,
                    shadowRadius: 20,
                  }}
                >
                  <CameraView 
                    ref={cameraRef} 
                    style={{ 
                      width: 320, 
                      height: 320,
                    }}
                    facing="front"
                    mirror={true}
                  />
                </View>

                {/* Dotted Circle Border Overlay */}
                <View 
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: 320,
                    height: 320,
                    borderRadius: 160,
                    borderWidth: 4,
                    borderColor: isMatched ? '#10B981' : '#34D399',
                    borderStyle: 'dashed',
                  }}
                />

                {/* Scanning Animation Overlay */}
                {isComparing && (
                  <Animated.View 
                    style={{ 
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      width: 320,
                      height: 2,
                      backgroundColor: '#10B981',
                      transform: [{ translateY: scanTranslateY }],
                      shadowColor: '#10B981',
                      shadowOffset: { width: 0, height: 0 },
                      shadowOpacity: 0.8,
                      shadowRadius: 10,
                    }} 
                  />
                )}

                {/* Match Status Indicator */}
                {isMatched && (
                  <View style={{
                    position: 'absolute',
                    top: -12,
                    right: -12,
                    backgroundColor: '#10B981',
                    borderRadius: 30,
                    padding: 12,
                    shadowColor: '#10B981',
                    shadowOffset: { width: 0, height: 4 },
                    shadowOpacity: 0.3,
                    shadowRadius: 8,
                  }}>
                    <Text style={{ color: 'white', fontSize: 24, fontWeight: 'bold' }}>✓</Text>
                  </View>
                )}
              </View>
            </View>

            {/* Status Message Below Circle */}
            {isMatched && (
              <View className="mb-6">
                <Text className="text-green-600 text-center text-lg font-bold mb-1">
                  ✅ Identity Verified!
                </Text>
                <Text className="text-gray-600 text-center text-sm">
                  Match confidence: {confidence.toFixed(1)}%
                </Text>
              </View>
            )}
          </View>

          {/* Bottom Controls */}
          <View className="pb-8 px-6 bg-white">
            <View className="flex-row gap-4">
              <TouchableOpacity
                onPress={() => {
                  setIdUploaded(false);
                  setIsMatched(false);
                  setMatchStatus("Position your face");
                  setConfidence(0);
                }}
                className="flex-1 bg-gray-200 px-6 py-4 rounded-xl active:bg-gray-300"
              >
                <Text className="text-gray-800 text-center text-base font-semibold">
                  Cancel
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                disabled={!isMatched}
                className={`flex-1 px-6 py-4 rounded-xl ${
                  isMatched ? 'bg-blue-600 active:bg-blue-700' : 'bg-gray-300'
                }`}
              >
                <Text className={`text-center text-base font-semibold ${
                  isMatched ? 'text-white' : 'text-gray-500'
                }`}>
                  Subscribe
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}
    </View>
  );
}