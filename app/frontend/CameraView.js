import { Camera, CameraType } from 'expo-camera';
import { useState, useRef } from 'react';
import { Image, Button, StyleSheet, Text, TouchableOpacity, View, useWindowDimensions  } from 'react-native';



export default function CameraView() {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [photoUri, setPhotoUri] = useState(null);

  // Calculate camera height based on screen width (Assumes 16:9 aspect ratio)
  const {width} = useWindowDimensions();
  const camera_height = Math.round((width * 16) / 9);
  
  const cameraRef = useRef(null);

  if (!permission) {
    // Camera permissions are still loading
    return <View/>;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>Allow access to camera to continue.</Text>
        <Button onPress={requestPermission} title="Allow" />
      </View>
    );
  }

  async function takePicture() {
    console.log("Taking picture...");

    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      setPhotoUri(photo.uri); // Save the photo URI to state
      
      // Create a FormData object to send the image to the server
      let formData = new FormData();
  
      formData.append('image', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });
  
      // Send the FormData object to the server
      fetch('/image', {
        method: 'POST',
        body: formData,
        headers: {
          'content-type': 'multipart/form-data',
        },
      });
    }
  }

  function flipCamera() {
    setType(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }
  return (
    <View style={{ flex: 1 }}>
      <Camera ref={cameraRef} ratio="16:9" style={{height: camera_height}} type={type}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={flipCamera}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={takePicture}>
            <Text style={styles.text}>Take Picture</Text>
          </TouchableOpacity>
        </View>
      </Camera>
    {photoUri && <Image source={{ uri: photoUri }} style={{ width: 200, height: 200 }} />}
  </View>
  )
}

const styles = StyleSheet.create({
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});