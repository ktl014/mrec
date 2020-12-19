from mrec.visualization.SessionState import SessionState

class TestSessionState():
    def test_SessionState(self):
        session_state = SessionState(user_name='', favorite_color='black')
        assert session_state.favorite_color == 'black'

    # def test_get(self):
    #     session_state = get(user_name='Mary', favorite_color='black')
    #     assert 'Mary' == session_state.user_name
