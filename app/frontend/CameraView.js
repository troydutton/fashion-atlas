import { Camera, CameraType } from 'expo-camera';
import { useState, useRef } from 'react';
import { Button, Text, TouchableOpacity, View} from 'react-native';
import { useIsFocused  } from '@react-navigation/native';
import Icon from 'react-native-ico-material-design';
import { styles } from './styles/style';

export default function CameraView({ navigation }) {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const isFocused = useIsFocused()
  const cameraRef = useRef(null);

  if (!permission) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionTextContainer}>
          <Text style={styles.text}>Allow access to camera to continue.</Text>
        </View>
        <Button onPress={requestPermission} title="Allow" />
      </View>
    )
  }

  async function takePicture() {
    console.log("Taking picture...");

    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();

      // Create a FormData object to send the image to the server
      let formData = new FormData();

      formData.append('image', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });

      // Send the FormData object to the server
      fetch('http://100.110.148.13:5000/image', {
        method: 'POST',
        body: formData,
        headers: {
          'content-type': 'multipart/form-data',
        },
      }).then(response => {
        if (response.status === 400) {
          throw new Error('No image provided');
        } else if (response.status === 500) {
          throw new Error('No detections');
        } else if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      }).then(data => {
        // navigation.navigate('Images', {image: photo.uri, detections: data});
        navigation.navigate('Images', {similarImages: data[0].similar_images});
      }).catch(error => {
        // Handle the error
        console.error(error);
      });
    }
  }

  function flipCamera() {
    setType(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }
  return (
    <View style={styles.container}>
        {isFocused && <Camera ref={cameraRef} ratio="16:9" style={styles.camera} type={type} />}
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
