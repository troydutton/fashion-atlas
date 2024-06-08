import { TouchableOpacity, View, ImageBackground , DeviceEventEmitter } from 'react-native';
import Icon from 'react-native-ico-material-design';
import { styles } from './styles/style';

export default function SelectionView({ navigation, route}) {
    const detections = route.params.detections;

    console.log(detections[0].bounding_box[2] - detections[0].bounding_box[0])
    return (
      <View style={styles.container}>
        <View style={{marginTop: 20, padding: 20}}>
            <TouchableOpacity onPress={() => navigation.navigate('Camera')}>
                <Icon name="go-back-left-arrow" color="white" width="40" height="40"/>
            </TouchableOpacity>
        </View>
        <ImageBackground  source={{uri: route.params.image}} style={styles.image}>
            {detections.map((detection, index) => (
                <View
                    key={index}
                    style={{
                        borderWidth: 2,
                        borderColor: 'red',
                        position: 'absolute',
                        left: detection.bounding_box[0],
                        top: detection[1],
                        width: detection.bounding_box[2] - detection.bounding_box[0],
                        height: detection.bounding_box[3] - detection.bounding_box[1],
                    }}
                />
            ))}
        </ImageBackground >
      </View>
    );
  }