import React, { useState, useEffect, useRef } from 'react'
import { StyleSheet, Text, View, Pressable, ImageBackground, Dimensions } from 'react-native'
import { Camera } from 'expo-camera'
import { manipulateAsync, FlipType, SaveFormat } from 'expo-image-manipulator'
import * as tf from '@tensorflow/tfjs'
import * as jpeg from 'jpeg-js'
import { Buffer } from 'buffer'

// Must import & override fetch otherwise will get "FileReader.readAsArrayBuffer is not implemented."
// Causes "Platform browser has already been set. Overwriting the platform with [object Object]."
import '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
// import * as cocossd from '@tensorflow-models/coco-ssd'

export default function App() {

  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
	const [cameraReady, setCameraReady] = useState(false)
	const [tfReady, setTfReady] = useState(false)
	const camera = useRef(null)
	const [photo, setPhoto] = useState(null)
	const [sm, setSm] = useState(null)
	const [mm, setMm] = useState(null)
	const [sm_pred, setSmPred] = useState([])
	const [mm_pred, setMmPred] = useState([])
	const [narration, setNarration] = useState('')
	const windowheight = Dimensions.get('window').height
	const windowwidth = Dimensions.get('window').width

	// Request permissions
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync()
      setHasPermission(status === 'granted')
    })()
		setPhoto(null)
  }, [])

	// Load TensorFlow + models
	useEffect( async () => {
		// tf.setBackend('cpu').then(
			tf.ready().then(async () => {
				setTfReady(true)
				mobilenet.load().then(setSm)
				// cocossd.load().then(setMm)
			})
		// )
		
	}, [])

	useEffect(async () => {
		if (photo && tfReady) {

			setNarration('Photo taken')
			let image = jpeg.decode(
				Buffer.from(
					photo.base64, 
					'base64'
				), 
				{ useTArray: true, formatAsRGBA: false }
			)

			setNarration('Image decoded')

			let tensor = tf.tensor3d(
				image.data, 
				[image.height, image.width, 3]
			)

			setNarration('Converted to tensor')

			if (sm) {
				setSmPred(await sm.classify(tensor))
			}

			// if (mm) {
			// 	mm.detect(tensor).then(setMmPred)
			// }

			setNarration('Predicting...')

		}
	}, [ photo ])

	useEffect(() => {
		console.log(sm_pred)
	}, [sm_pred])

	// Deal with situations surrounding no / awaiting permissions
  if (hasPermission === null) {
    return <View />
  }

  if (hasPermission === false) {
    return <View style={styles.middletext}>
			<Text>No access to camera</Text>
		</View>
  }

	// GUI
  return (
    <>
			{tfReady && sm ? 
				<View style={styles.container}>
					{photo ? 
						<Pressable
							onPressOut={() => {
								setPhoto(null)
								setNarration('')
								setSmPred([])
								setMmPred([])
							}}
						>

							<ImageBackground
								style={{ width: windowwidth, height: windowheight}}
								source={photo}
							>
								<View style={styles.middletext}>
									{sm_pred.length ? 
									sm_pred.map(e => <Text 
										style={{ color: 'white'}}
										key={e.className}
									>{e.className + ' ' + e.probability}</Text>)
									: <Text style={{ color: 'white'}}>{narration}</Text>}
								</View>
							</ImageBackground>
						</Pressable>
						:
						<>
							<Camera 
								style={styles.camera} 
								type={type}
								onCameraReady={() => setCameraReady(true)}
								ref={camera}
							>
								<View style={styles.buttonContainer}>
									<Pressable
										style={styles.button}
										onPressOut={() => {
											setType(
												type === Camera.Constants.Type.back
													? Camera.Constants.Type.front
													: Camera.Constants.Type.back
											);
										}}
									>
										<Text style={styles.text}> Flip </Text>
									</Pressable>

									<Pressable
										style={styles.button}
										onPressOut={async () => {
											if (cameraReady) {
												camera.current.takePictureAsync(
													{quality: 1}
												).then(p => {
													if (type === Camera.Constants.Type.front) {
														manipulateAsync(
															p.uri, 
															[
																{ rotate: 180 }, 
																{ flip: FlipType.Vertical }, 
																{ crop: { 
																	originX: 0, 
																	originY: 0, 
																	width: p.width - 1, 
																	height: p.height - 1}
																}, 
																{ resize: { width: windowwidth } }
															],
															{ base64: true, compress: 1, format: SaveFormat.JPEG }
														).then(setPhoto)
													} else {
														manipulateAsync(
															p.uri, 
															[{ resize: { width: windowwidth } }],
															{ base64: true, compress: 1, format: SaveFormat.JPEG }
														).then(setPhoto)
													}
												})
											}
										}}
									>
										<Text style={styles.text}> Clip </Text>
									</Pressable>
								</View>
							</Camera>
						</>
					}
				</View>
			:
			<View style={styles.middletext}>
				<Text>
					Loading...
				</Text>
			</View>
			}
    </>
  )
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
		padding: 50,
  },
  text: {
    fontSize: 18,
    color: 'white',
  },
	middletext: {
		flex: 1,
		// backgroundColor: '#fff',
		alignItems: 'center',
		justifyContent: 'center',
	}
});