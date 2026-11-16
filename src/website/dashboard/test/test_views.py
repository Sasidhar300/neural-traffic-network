from http import HTTPStatus
from copy import deepcopy
from unittest.mock import patch

from django.test import TestCase
from django.contrib.auth.models import User
from dashboard.views import SIM_CONFIG
from .utils import *


class TestDashboardViews(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username=TEST_USER, password=TEST_PASSWORD)

    def test_home_view_without_user(self):
        response = self.client.get(HOME_URL)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, HOME_TEMPLATE)
        self.assertFalse(response.context['user'].is_authenticated)
        self.assertContains(response, "Welcome to Neural Traffic Networks")
        self.assertContains(response, "Sign In")
        self.assertNotContains(response, "Sign out")
        self.assertNotContains(response, "Get Started")

    def test_home_view_with_user(self):
        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        response = self.client.get(HOME_URL)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, HOME_TEMPLATE)
        self.assertTrue(response.context['user'].is_authenticated)
        self.assertContains(response, "Welcome to Neural Traffic Networks")
        self.assertNotContains(response, "Sign In")
        self.assertContains(response, "Sign out")
        self.assertContains(response, "Get Started")

    def test_register_view_when_logged_out(self):
        response = self.client.get(REGISTER_URL)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertFalse(response.context['user'].is_authenticated)
        self.assertTemplateUsed(response, REGISTER_TEMPLATE)
        self.assertContains(response, "Register")
        self.assertNotContains(response, "Sign out")

    def test_register_view_with_user(self):
        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        response = self.client.get(REGISTER_URL)

        self.assertRedirects(response, HOME_URL)

        followup_response = self.client.get(HOME_URL)
        self.assertTrue(followup_response.context['user'].is_authenticated)

    def test_signin_view_when_logged_out(self):
        response = self.client.get(SIGNIN_URL)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertFalse(response.context['user'].is_authenticated)
        self.assertTemplateUsed(response, SIGNIN_TEMPLATE)
        self.assertContains(response, "Sign In")
        self.assertNotContains(response, "Sign out")

    def test_signin_view_with_user(self):
        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        response = self.client.get(SIGNIN_URL)

        self.assertRedirects(response, HOME_URL)

        followup_response = self.client.get(HOME_URL)
        self.assertTrue(followup_response.context['user'].is_authenticated)

    def test_signin_view_with_invalid_credentials(self):
        response = self.client.post(SIGNIN_URL, {
            USERNAME_KEY: INVALID_USERNAME_KEY,
            PASSWORD_KEY: INVALID_PASSWORD_KEY
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, SIGNIN_TEMPLATE)
        self.assertContains(response, "Invalid username or password.")
        self.assertFalse(response.context['user'].is_authenticated)

    def signout_view_when_logged_out(self):
        response = self.client.get(SIGNOUT_URL)

        self.assertRedirects(response, HOME_URL)
        followup_response = self.client.get(HOME_URL)
        self.assertFalse(followup_response.context['user'].is_authenticated)

    def test_signout_view_with_user(self):
        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        response = self.client.get(SIGNOUT_URL)

        self.assertRedirects(response, SIGNIN_URL)

        followup_response = self.client.get(SIGNIN_URL)
        self.assertFalse(followup_response.context['user'].is_authenticated)

    def test_register_view_post(self):
        """
        Test the registration process. (Positive case)
        """

        response = self.client.get(REGISTER_URL)
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, REGISTER_TEMPLATE)

        response = self.client.post(REGISTER_URL, {
            USERNAME_KEY: NEW_TEST_USER,
            PASSWORD_KEY: NEW_TEST_PASSWORD,
            CONFIRM_PASSWORD_KEY: NEW_TEST_PASSWORD,
            EMAIL_KEY: NEW_TEST_EMAIL
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, "Account created successfully.")
        self.assertFalse(response.context['user'].is_authenticated)

        # Verify if user created
        user = User.objects.get(username=NEW_TEST_USER)
        self.assertIsNotNone(user)
        self.assertTrue(user.check_password(NEW_TEST_PASSWORD))
        self.assertEqual(user.email, NEW_TEST_EMAIL)

    def test_register_view_post_username_already_exists(self):
        """
        Test the registration process. (negative case)
        when the user already exists.
        """
        self.test_register_view_post()

        # Attempt to register the same user again
        response = self.client.post(REGISTER_URL, {
            USERNAME_KEY: NEW_TEST_USER,
            PASSWORD_KEY: NEW_TEST_PASSWORD,
            CONFIRM_PASSWORD_KEY: NEW_TEST_PASSWORD,
            EMAIL_KEY: NEW_TEST_EMAIL
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, "Username already exists.")

    def test_register_view_post_email_already_exists(self):
        """
        Test the registration process. (negative case)
        when the user already exists.
        """
        self.test_register_view_post()

        # Attempt to register the same user again
        response = self.client.post(REGISTER_URL, {
            USERNAME_KEY: INVALID_USERNAME_KEY,
            PASSWORD_KEY: NEW_TEST_PASSWORD,
            CONFIRM_PASSWORD_KEY: NEW_TEST_PASSWORD,
            EMAIL_KEY: NEW_TEST_EMAIL
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, "Email is already registered.")


    def test_register_view_post_password_mismatch(self):
        """
        Test the registration process. (negative case)
        when the password and confirm password do not match.
        """
        response = self.client.post(REGISTER_URL, {
            USERNAME_KEY: NEW_TEST_USER,
            PASSWORD_KEY: NEW_TEST_PASSWORD,
            CONFIRM_PASSWORD_KEY: NEW_INCORRECT_PASSWORD,
            EMAIL_KEY: NEW_TEST_EMAIL
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, "Passwords do not match.")

        self.assertFalse(User.objects.filter(username=NEW_TEST_USER).exists())


    def test_register_view_post_empty_fields(self):
        """
        Test the registration process. (negative case)
        when the fields are empty.
        """
        response = self.client.post(REGISTER_URL, {
            USERNAME_KEY: EMPTY_STRING,
            PASSWORD_KEY: EMPTY_STRING,
            CONFIRM_PASSWORD_KEY: EMPTY_STRING,
            EMAIL_KEY: NEW_TEST_EMAIL
        })

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, "Required fields are missing.")
        self.assertFalse(User.objects.filter(username=NEW_TEST_USER).exists())

    def test_about_view(self):
        """
        Test the about view.
        """
        response = self.client.get(ABOUT_URL)
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, ABOUT_TEMPLATE)

    def test_simulation_view_when_logged_off(self):
        """
        Test the simulation view. (negative case)
        """
        response = self.client.get(SIMULATION_URL)
        self.assertRedirects(response, SIGNIN_URL + NEXT + SIMULATION_URL)

    def test_simulation_view_when_logged_in(self):
        """
        Test the simulation view. (positive case)
        """
        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        response = self.client.get(SIMULATION_URL)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTemplateUsed(response, SIMULATION_TEMPLATE)
        self.assertContains(response, "Select Your")


    @patch('dashboard.views.threading.Thread')
    @patch('dashboard.views.simulation_tracker')
    def test_run_simulation_output_view(self, tracker, mock_thread_class):
        """
        Test the simulation output view. (positive case)
        """
        tracker.is_running.return_value = False
        tracker.start_simulation.return_value = True

        self.client.login(username=TEST_USER, password=TEST_PASSWORD)
        with patch('dashboard.views.SIM_CONFIG', deepcopy(SIM_CONFIG)):
            response = self.client.post(
                RUN_SIMULATION_URL, dict(RUN_SIMULATION_DATA, queue_penalty="3")
            )

        mock_thread_class.assert_called_once()
        mock_thread_class.return_value.start.assert_called_once()
        (simulation_config,) = mock_thread_class.call_args.kwargs["args"]
        self.assertEqual(simulation_config.reward_params.queue_penalty, 3)

        self.assertRedirects(response, SIMULATION_OUTPUT_URL, fetch_redirect_response=False)
