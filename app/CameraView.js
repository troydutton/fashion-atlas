import { Camera, CameraType } from 'expo-camera/legacy';
import { useState, useRef } from 'react';
import { Button, Text, TouchableOpacity, View, ActivityIndicator} from 'react-native';
import { useIsFocused  } from '@react-navigation/native';
import Icon from 'react-native-ico-material-design';
import { styles } from './styles/style';

export default function CameraView({ navigation }) {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [isLoading, setIsLoading] = useState(false);
  const isFocused = useIsFocused()
  const cameraRef = useRef(null);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionTextContainer}>
          <Text style={styles.text}>Accept camera permissions to continue.</Text>
        </View>
        <Button onPress={requestPermission} title="Accept Camera Permissions"/>
      </View>
    )
  }

  async function takePicture() {
    console.log("Taking picture...");

    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();

      let formData = new FormData();

      formData.append('image', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });

      setIsLoading(true);

      fetch('http://100.110.148.16:5000/image', {
        method: 'POST',
        body: formData,
        headers: {
          'content-type': 'multipart/form-data',
        },
      }).then(response => {
        setIsLoading(false);

        if (response.status === 400) {
          throw new Error('No image provided');
        } else if (response.status === 500) {
          throw new Error('No detections');
        } else if (!response.ok) {
          throw new Error(`HTTP error status: ${response.status}`);
        }
        return response.json();
      }).then(data => {
        navigation.navigate('Selection', {imageUri: photo.uri, detections: data});
      }).catch(error => {
        console.error(error);
      });
    }
  }

  function flipCamera() {
    setType(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="black" />
        <Text color="black">Detecting Garments...</Text>
      </View>
    );
  } else {
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
}
