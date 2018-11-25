from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
	comment=StringField('Enter User ID',validators=[DataRequired()])
	submit=SubmitField('Lets eat!')