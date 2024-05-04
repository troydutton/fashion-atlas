import { TouchableOpacity, View, Image, FlatList } from 'react-native';
import Icon from 'react-native-ico-material-design';
import { styles } from './styles/style';

export default function SelectionView({ navigation, route}) {
    const similarImages = route.params.similarImages;

    const renderItem = ({ item, index }) => (
        <Image
            key={index}
            style={styles.image}
            source={{uri: `data:image/jpeg;base64,${item}`}}
        />
    );

    return (
        <View style={ styles.container}>
            <View style={{marginTop: 50, padding: 5}}>
                <TouchableOpacity onPress={() => navigation.navigate('Camera')}>
                    <Icon name="go-back-left-arrow" color="white" width="40" height="40"/>
                </TouchableOpacity>
            </View>
            <FlatList
                data={similarImages}
                renderItem={renderItem}
                keyExtractor={(item, index) => index.toString()}
                numColumns={2}
            />
        </View>
    );
  }