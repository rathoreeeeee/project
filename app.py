import os
import json
import numpy as np
import music21 as m21
import streamlit as st
import tensorflow.keras as keras


MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64



class MelodyGenerator:

    def __init__( self, model_path = "model.keras" ):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)


        with open(MAPPING_PATH , "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbol = ["/"] * SEQUENCE_LENGTH



    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbols
        seed = seed.split()
        melody = seed  
        seed = self._start_symbol + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]



        for _ in range( num_steps ):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length :]


            #one-hot encode the seed
            onehot_seed = keras.utils.to_categorical( seed, num_classes=len(self._mappings)+1)
            
            # currently the onehot_seed is a 2d array having shape : ( max_sequence_length , num of symbol in  the vocabulary )
            # but generaly model.predict accepts a 3d array beacause it calculates probability for bunch of data not for 1 so . we are adding a extra dimention
            onehot_seed = onehot_seed[np.newaxis , ...]

            # make a prediction

            probabilities = self.model.predict( onehot_seed )[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1

            output_int = self._sample_with_temperature(probabilities , temperature)

            #update the seed
            seed.append(output_int)

            # map int to our encodeing
            output_symbol =  [ k for k,v in self._mappings.items()  if v == output_int ][0]

            # check whether we'r at the end of a melody
            if output_symbol == "/":
                break

            # update the melody
            else :
                melody.append(output_symbol)




        return melody       



    def _sample_with_temperature( self, probabilities , temperature ):

        # temperature -> infinity
        # temperature -> 0
        # temperature = 1

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp( predictions ) / np.sum(np.exp(predictions))


        choices = range(len(probabilities))
        index = np.random.choice(choices , p = probabilities)

        return index
    


    def save_melody( self, melody ,step_duration = 0.25, format = "midi" , file_name = "mel.mid") :

        #create a music21 stream
        stream = m21.stream.Stream()


        #parse all the symbol in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1


        for  i, symbol in enumerate(melody) :

            #handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                #ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest( quarterLength = quarter_length_duration)

                    #hadle notes
                    else :
                        m21_event = m21.note.Note( int(start_symbol) , quaterLegth = quarter_length_duration )   


                    stream.append(m21_event)    
                    
                    # reset the step counter
                    step_counter = 1


                start_symbol = symbol        


            # handle case in which we have a prolongation sign "_"
            else :
                step_counter = step_counter + 1



        # write the m21 strem to a midi file
        stream.write( format , file_name )

   

# app initiallize 

# st.title('Melody Craft 1.0')

st.markdown(
        """
        <style>
        /* Define keyframes for the animation */
        @keyframes glow {
            0% { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #fff, 0 0 40px #ff00e4, 0 0 70px #ff00e4, 0 0 80px #ff00e4, 0 0 100px #ff00e4, 0 0 150px #ff00e4; }
            50% { text-shadow: 0 0 20px #fff, 0 0 30px #fff, 0 0 40px #ff00e4, 0 0 50px #ff00e4, 0 0 80px #ff00e4, 0 0 90px #ff00e4, 0 0 110px #ff00e4, 0 0 160px #ff00e4; }
            100% { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #fff, 0 0 40px #ff00e4, 0 0 70px #ff00e4, 0 0 80px #ff00e4, 0 0 100px #ff00e4, 0 0 150px #ff00e4; }
        }

        /* Apply animation to the title */
        .glowing-title {
            font-size: 6rem;
            font-weight: bold;
            color: #ff00e4;
            animation: glow 2s ease-in-out infinite;
        }
        </style>
        """
        , unsafe_allow_html=True
    )
st.markdown('<h1 class="glowing-title">Tune Genie</h1>', unsafe_allow_html=True)



seeds ={  'seed1' : "60 _ _ _ _ _ _ _ 60 _ _ _ _ _ _ _ 60 _ _ _ _ _ _ _ 62",'seed2' : "69 _ _ _ _ _ _ _",'seed3' : "69 _ _ _ 69 _ _ _ 69 _ _ _ 72 _ _ _ 72 _ _ _ 69",'seed4' : "76 _ _ _ 76 _ _ _ _ _ 76 _ 76 _ _ _ 79 _ _ _ 74 ",'seed5': "71 _ _ _ 69 _" }


seed = st.selectbox( 'choose seed ' , ('seed1','seed2','seed3','seed4','seed5','Try your own seed' ))


if seed == "Try your own seed":
    seed = st.text_input("Enter your custom seed:")
else:
    seed = seeds[seed]    


if st.button('GENERATE'):
    mg = MelodyGenerator()
    melody = mg.generate_melody( seed , 200, SEQUENCE_LENGTH , 0.4)
    print(melody)
    mg.save_melody(melody)
    st.download_button(
            label="Download Midi File",
            data=open("mel.mid", "rb").read(),
            file_name="example.midi",
            mime="audio/midi"
        )
