import React from 'react';
import styled from 'styled-components';
import Header from './components/Header';
import NewsAnalysis from './components/NewsAnalysis';

const AppContainer = styled.div`
  font-family: 'Noto Sans KR', sans-serif;
  background-color: white;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

function App() {
  return (
    <AppContainer>
      <Header />
      <NewsAnalysis />
    </AppContainer>
  );
}

export default App;