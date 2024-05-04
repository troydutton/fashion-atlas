import { Camera, CameraType } from 'expo-camera';
import { useState, useRef } from 'react';
import { Image, Button, StyleSheet, Text, TouchableOpacity, View, useWindowDimensions  } from 'react-native';
import Icon from 'react-native-ico-material-design';

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
      fetch('http://100.110.148.60:5000/image', {
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
    <View style={styles.container}>
      <Camera ref={cameraRef} ratio="16:9" style={{height: camera_height, marginTop: 45}} type={type} />
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.flipButton} onPress={flipCamera}>
          <Icon name="synchronization-button-with-two-arrows" color="white" width="60" height="60"/>
        </TouchableOpacity>
        <TouchableOpacity style={styles.takePictureButton} onPress={takePicture}>
          <Icon name="radio-on-button" color="white" width="70" height="70"/>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black', // Set the background color to black
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center', // Center the buttons horizontally
    backgroundColor: 'transparent',
    padding: 20, // Add some padding if needed
  },
  flipButton: {
    alignSelf: 'center',
    position: 'absolute', // Position the flip button absolutely
    left: 40, // Position it a little to the left
  },
  takePictureButton: {
    alignSelf: 'center', // Center the take picture button vertically
  }
});