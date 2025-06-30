"""
Authentication Forms using Flask-WTF
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
# NEW import in forms.py
from app.models_new.user_model import User

class LoginForm(FlaskForm):
    """Login form"""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required'),
        Length(min=3, max=20, message='Username must be between 3 and 20 characters')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required')
    ])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')


class RegistrationForm(FlaskForm):
    """User registration form"""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required'),
        Length(min=3, max=20, message='Username must be between 3 and 20 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required'),
        Length(min=6, message='Password must be at least 6 characters long')
    ])
    password2 = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password'),
        EqualTo('password', message='Passwords must match')
    ])
    role = SelectField('Role', choices=[
        ('viewer', 'Viewer - View dashboards only'),
        ('analyst', 'Analyst - Upload data and view models')
    ], validators=[DataRequired()])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        """Check if username is already taken"""
        user = User.get_by_username(username.data)
        if user:
            raise ValidationError('Username already taken. Please choose a different username.')
    
    def validate_email(self, email):
        """Check if email is already registered"""
        user = User.get_by_email(email.data)
        if user:
            raise ValidationError('Email already registered. Please use a different email address.')


class ChangePasswordForm(FlaskForm):
    """Change password form"""
    current_password = PasswordField('Current Password', validators=[
        DataRequired(message='Current password is required')
    ])
    new_password = PasswordField('New Password', validators=[
        DataRequired(message='New password is required'),
        Length(min=6, message='Password must be at least 6 characters long')
    ])
    new_password2 = PasswordField('Confirm New Password', validators=[
        DataRequired(message='Please confirm your new password'),
        EqualTo('new_password', message='Passwords must match')
    ])
    submit = SubmitField('Change Password')


class ProfileForm(FlaskForm):
    """User profile form"""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required'),
        Length(min=3, max=20, message='Username must be between 3 and 20 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    submit = SubmitField('Update Profile')
    
    def __init__(self, original_username=None, original_email=None, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email
    
    def validate_username(self, username):
        """Check if username is already taken (excluding current user)"""
        if username.data != self.original_username:
            user = User.get_by_username(username.data)
            if user:
                raise ValidationError('Username already taken. Please choose a different username.')
    
    def validate_email(self, email):
        """Check if email is already registered (excluding current user)"""
        if email.data != self.original_email:
            user = User.get_by_email(email.data)
            if user:
                raise ValidationError('Email already registered. Please use a different email address.')


class AdminUserForm(FlaskForm):
    """Admin form for managing users"""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required'),
        Length(min=3, max=20, message='Username must be between 3 and 20 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Please enter a valid email address')
    ])
    role = SelectField('Role', choices=[
        ('viewer', 'Viewer'),
        ('analyst', 'Analyst'),
        ('admin', 'Admin')
    ], validators=[DataRequired()])
    submit = SubmitField('Update User')