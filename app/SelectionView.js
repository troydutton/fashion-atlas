import { TouchableOpacity, View, ActivityIndicator, Text, ImageBackground , Dimensions } from 'react-native';
import Icon from 'react-native-ico-material-design';
import { styles } from './styles/style';

export default function SelectionView({ navigation, route}) {
    const detections = route.params.detections;
    const imageUri = route.params.imageUri;

    const colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'pink', 'lime'];

    console.log(detections[0].bounding_box)
    
    // TODO: Hardcoded aspect ratio
    const imgWidth = Dimensions.get('window').width;
    const imgHeight = imgWidth * (16/9);

    console.log(imgWidth)
    console.log(imgHeight)
    
    return (
      <View style={styles.container}>
        <View style={{marginTop: 20, padding: 20}}>
            <TouchableOpacity onPress={() => navigation.navigate('Camera')}>
                <Icon name="go-back-left-arrow" color="white" width="40" height="40"/>
            </TouchableOpacity>
        </View>
        <ImageBackground source={{uri: imageUri}} style={styles.fullscreenImage}>
            {detections.map((detection, index) => (
                <TouchableOpacity
                    key={index}
                    onPress={() => navigation.navigate('Images', {
                        imageUri: imageUri,
                        detections: detections,
                        similarGarments: detection.similar_garments,
                        similarModels: detection.similar_models
                    })}                  
                    style={{
                        borderWidth: 2,
                        borderColor: colors[index % colors.length],
                        position: 'absolute',
                        left: detection.bounding_box[0] * imgWidth,
                        top: detection.bounding_box[1] * imgHeight,
                        width: (detection.bounding_box[2] - detection.bounding_box[0]) * imgWidth,
                        height: (detection.bounding_box[3] - detection.bounding_box[1]) * imgHeight,
                    }}
                >
                    <Text style={{color: colors[index % colors.length], fontSize: 10, top:-20}}>
                        {detection.class_name}
                    </Text>
                </TouchableOpacity>
            ))}
        </ImageBackground >
      </View>
    );
  }